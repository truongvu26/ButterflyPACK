#pragma once
#ifdef H2_HAVE_GPU
#include "../../GPU_INTERFACE/bpack_gpu.h"
#include "device_heap.hpp"
#include "evaluator_device.hpp"
#include <chrono>
#include <climits>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace fmm { namespace gpu {
// Collective HOST preparation is a separate evaluator form, not a GPU block
// callback. Native construction guarantees matching entry phases on comm_.
class HostPanelPreparation {
public:
    HostPanelPreparation(bpack_gpu_host_block_callback callback,void* user,MPI_Comm comm,
                         size_t budget,int column_base)
      : callback_(callback),user_(user),comm_(comm),budget_(budget),column_base_(column_base) {
        if (!callback || comm==MPI_COMM_NULL || budget<sizeof(dcomplex) || budget>size_t(INT_MAX))
            throw std::invalid_argument("GPU host panels need callback, front communicator and a 16..INT_MAX byte budget");
        if (const char* checks=std::getenv("BPACK_CHECK"))
            if (*checks) throw std::invalid_argument("Collective host panels require BPACK_CHECK disabled");
    }
    const bpack_gpu_host_panel_stats& stats() const { return stats_; }
    template<typename T>
    void eval(const std::vector<EvalItemT<T>>& items,const char* meta,
              const PointTable& points,cudaStream_t stream) {
        const auto started=clock::now(); ++stats_.calls;
        auto& heap=DeviceHeap::instance();
        std::vector<BlockGatherItem> gather; int64_t nids=0; int maximum=0;
        for (const auto& x:items) {
            gather.push_back({x.rows,x.cols,x.m,x.n,nids,nids+x.m}); nids+=int64_t(x.m)+x.n;
            maximum=std::max(maximum,std::max(x.m,x.n));
        }
        std::vector<int64_t> ids(static_cast<size_t>(nids));
        if (!items.empty()) {
            auto* dg=heap.alloc<BlockGatherItem>(gather.size()*sizeof(BlockGatherItem));
            auto* di=heap.alloc<int64_t>(std::max<size_t>(ids.size(),1)*sizeof(int64_t));
            check_cuda(cudaMemcpyAsync(dg,gather.data(),gather.size()*sizeof(BlockGatherItem),cudaMemcpyHostToDevice,stream),"host panel gather metadata");
            launch_gather_block_points(dg,static_cast<int>(gather.size()),maximum,meta,points,di,nullptr,stream);
            check_cuda(cudaGetLastError(),"host panel gather ids");
            check_cuda(cudaMemcpyAsync(ids.data(),di,ids.size()*sizeof(int64_t),cudaMemcpyDeviceToHost,stream),"host panel ids");
            check_cuda(cudaStreamSynchronize(stream),"host panel ids ready");
            heap.free(di); heap.free(dg);
        }
        struct Tile { size_t item; int r,c,m,n; size_t bytes; };
        std::vector<std::vector<Tile>> windows; size_t used=0;
        const size_t entries=budget_/sizeof(T);
        for (size_t k=0;k<items.size();++k) {
            const auto& x=items[k];
            if (x.m<=0 || x.n<=0 || x.ld<x.m) throw std::invalid_argument("Invalid host panel output shape");
            const int tm=static_cast<int>(std::min<size_t>(x.m,entries));
            const int tn=static_cast<int>(std::min<size_t>(x.n,entries/tm));
            for (int c=0;c<x.n;c+=tn) for (int r=0;r<x.m;r+=tm) {
                const int m=std::min(tm,x.m-r),n=std::min(tn,x.n-c);
                const size_t bytes=size_t(m)*n*sizeof(T);
                if (windows.empty() || used+bytes>budget_) { windows.emplace_back(); used=0; }
                windows.back().push_back({k,r,c,m,n,bytes}); used+=bytes;
            }
        }
        int nranks=0; MPI_Comm_size(comm_,&nranks);
        int rounds=static_cast<int>(windows.size());
        MPI_Allreduce(MPI_IN_PLACE,&rounds,1,MPI_INT,MPI_MAX,comm_);
        stats_.metadata_seconds+=elapsed(started);
        for (int w=0;w<rounds;++w) {
            const auto mt=clock::now();
            std::vector<int> description,local_rows,local_cols;
            size_t local_entries=0;
            if (w<static_cast<int>(windows.size())) for (const auto& t:windows[w]) {
                description.push_back(t.m); description.push_back(t.n);
                for (int i=0;i<t.m;++i) local_rows.push_back(checked_id(ids[gather[t.item].row_at+t.r+i],0));
                for (int j=0;j<t.n;++j) local_cols.push_back(checked_id(ids[gather[t.item].col_at+t.c+j],column_base_));
                local_entries+=size_t(t.m)*t.n;
            }
            std::vector<int> desc_counts,row_counts,col_counts;
            auto desc=all_gather(description,desc_counts);
            auto rows=all_gather(local_rows,row_counts); auto cols=all_gather(local_cols,col_counts);
            int intersections=static_cast<int>(desc.size()/2),all_rows=static_cast<int>(rows.size()),all_cols=static_cast<int>(cols.size());
            std::vector<int> ms,ns,owners,pmaps(3*nranks,1);
            for (int p=0,offset=0;p<nranks;++p) {
                pmaps[2*nranks+p]=p;
                for (int i=0;i<desc_counts[p]/2;++i) {
                    ms.push_back(desc[offset++]); ns.push_back(desc[offset++]); owners.push_back(p);
                }
            }
            std::vector<T> values(std::max<size_t>(local_entries,1));
            int64_t local_count=static_cast<int64_t>(local_entries);
            stats_.metadata_seconds+=elapsed(mt);
            const auto pt=clock::now();
            callback_(&intersections,&all_rows,&all_cols,&local_count,rows.data(),cols.data(),values.data(),
                      ms.data(),ns.data(),owners.data(),&nranks,pmaps.data(),user_);
            stats_.preparation_seconds+=elapsed(pt);
            const auto ut=clock::now(); size_t at=0;
            if (w<static_cast<int>(windows.size())) for (const auto& t:windows[w]) {
                const auto& x=items[t.item];
                check_cuda(cudaMemcpy2DAsync(x.out+t.r+size_t(t.c)*x.ld,size_t(x.ld)*sizeof(T),values.data()+at,
                    size_t(t.m)*sizeof(T),size_t(t.m)*sizeof(T),t.n,cudaMemcpyHostToDevice,stream),"host panel upload");
                at+=size_t(t.m)*t.n;
            }
            // Host values cannot be resized or destroyed until all queued reads finish.
            check_cuda(cudaStreamSynchronize(stream),"host panel staging released");
            stats_.upload_seconds+=elapsed(ut); ++stats_.windows;
            stats_.upload_bytes+=local_entries*sizeof(T);
            stats_.peak_staging_bytes=std::max<uint64_t>(stats_.peak_staging_bytes,local_entries*sizeof(T));
        }
    }
private:
    using clock=std::chrono::steady_clock;
    static double elapsed(clock::time_point t) { return std::chrono::duration<double>(clock::now()-t).count(); }
    static int checked_id(int64_t id,int base) {
        id-=base;
        if (id<0 || id>=INT_MAX) throw std::out_of_range("Host panel application index outside 32-bit callback range");
        return static_cast<int>(id)+1;
    }
    std::vector<int> all_gather(const std::vector<int>& local,std::vector<int>& counts) {
        int p=0; MPI_Comm_size(comm_,&p); counts.resize(p);
        if (local.size()>size_t(INT_MAX)) throw std::overflow_error("Host panel metadata too large");
        int n=static_cast<int>(local.size()); MPI_Allgather(&n,1,MPI_INT,counts.data(),1,MPI_INT,comm_);
        std::vector<int> offsets(p); int64_t total=0;
        for (int i=0;i<p;++i) { offsets[i]=static_cast<int>(total); total+=counts[i]; if(total>INT_MAX) throw std::overflow_error("Host panel metadata too large"); }
        std::vector<int> global(static_cast<size_t>(total));
        MPI_Allgatherv(local.data(),n,MPI_INT,global.data(),counts.data(),offsets.data(),MPI_INT,comm_);
        return global;
    }
    bpack_gpu_host_block_callback callback_; void* user_; MPI_Comm comm_; size_t budget_; int column_base_;
    bpack_gpu_host_panel_stats stats_{};
};
}} // namespace fmm::gpu
#endif
