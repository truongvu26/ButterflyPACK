#pragma once
#ifdef H2_HAVE_GPU
#include "hodlr_device.hpp"
namespace bpack { namespace gpu {
template<typename T> class RectangularDevice {
public:
    RectangularDevice() = default;
    RectangularDevice(const RectangularDevice&) = delete;
    RectangularDevice& operator=(const RectangularDevice&) = delete;
    ~RectangularDevice() { release(); }
    int rows() const { return m_; }
    int cols() const { return n_; }
    int rank() const { return k_; }
    MPI_Comm comm() const { return comm_; }
    void upload(int m, int n, int k, const T* u, const T* v, MPI_Comm comm,
                const T* device_u=nullptr,const T* device_v=nullptr) {
        if (m < 0 || n < 0 || k <= 0 || comm == MPI_COMM_NULL)
            throw std::invalid_argument("Rectangular GPU dimensions/communicator");
        Context& ctx = Context::instance(); ctx.activate();
        release(); m_=m; n_=n; k_=k; comm_=comm;
        auto& heap=DeviceHeap::instance();
        // GPU BACA already owns mirrors in the enclosing GpuState. Borrow
        // those immutable buffers; the rectangular view dies before their
        // HodlrDevice owner. CPU-built factors instead get owned storage.
        if ((!m || device_u) && (!n || device_v)) {
            u_=const_cast<T*>(device_u); v_=const_cast<T*>(device_v); ready_=true;
            return;
        }
        const size_t count=(static_cast<size_t>(m)+n)*k;
        store_=heap.alloc_resident<T>(std::max<size_t>(count,1)*sizeof(T));
        u_=store_; v_=store_+static_cast<size_t>(m)*k;
        if (m) check_cuda(cudaMemcpyAsync(u_,u,static_cast<size_t>(m)*k*sizeof(T),cudaMemcpyHostToDevice,ctx.stream()),"rectangular U upload");
        if (n) check_cuda(cudaMemcpyAsync(v_,v,static_cast<size_t>(n)*k*sizeof(T),cudaMemcpyHostToDevice,ctx.stream()),"rectangular V upload");
        check_cuda(cudaStreamSynchronize(ctx.stream()),"rectangular upload");
        ready_=true;
    }
    void conjugate_in_place(T* a,int rows,int cols) {
        if constexpr (fmm::gpu::is_complex_scalar<T>) {
            auto& ctx=Context::instance(); ctx.activate();
            conjugate(a,a,rows,cols,ctx.stream());
        }
    }
    // Synchronous on the backend stream. Input/output are packed local arrays.
    // Only k*nrhs coefficients use host staging for non-GPU-aware MPI.
    double mult(char op, int nrhs, const T* x, T* y, int mode, bool device=false) {
        if (!ready_ || nrhs<0 || (op!='N' && op!='T' && op!='C'))
            throw std::invalid_argument("Rectangular GPU multiply state/op");
        auto& ctx=Context::instance(); ctx.activate();
        auto stream=ctx.stream(); auto& heap=DeviceHeap::instance();
        fmm::gpu::tensor_core_gemm()=(mode==2);
        const int ni=op=='N'?n_:m_, no=op=='N'?m_:n_;
        const size_t ib=static_cast<size_t>(ni)*nrhs*sizeof(T), ob=static_cast<size_t>(no)*nrhs*sizeof(T);
        T* dx=device?const_cast<T*>(x):heap.alloc<T>(std::max<size_t>(ib,sizeof(T)));
        T* input_storage=dx;
        T* dy=device?y:heap.alloc<T>(std::max<size_t>(ob,sizeof(T)));
        T* coeff=heap.alloc<T>(std::max<size_t>(static_cast<size_t>(k_)*nrhs,1)*sizeof(T));
        if (!device && ib) check_cuda(cudaMemcpyAsync(dx,x,ib,cudaMemcpyHostToDevice,stream),"rectangular input");
        // A^H x = conj(A^T conj(x)); private scratch permits alias-free C.
        T* conjugated=nullptr;
        if (op=='C' && fmm::gpu::is_complex_scalar<T>) {
            conjugated=heap.alloc<T>(std::max<size_t>(ib,sizeof(T)));
            conjugate(dx,conjugated,ni,nrhs,stream); dx=conjugated;
        }
        check_cuda(cudaMemsetAsync(coeff,0,static_cast<size_t>(k_)*nrhs*sizeof(T),stream),"rectangular coefficients");
        VBatch<T> reduce,apply;
        if (ni && nrhs) reduce.entries.push_back({op=='N'?v_:u_,dx,coeff,k_,nrhs,ni,ni,ni,k_});
        if (no && nrhs) apply.entries.push_back({op=='N'?u_:v_,coeff,dy,no,nrhs,k_,no,k_,no});
        meta_.clear(); reduce.stage(meta_); apply.stage(meta_);
        char* md=meta_.upload(md_,stream);
        reduce.gemm(md,MagmaTrans,MagmaNoTrans,T(1),T(0),ctx.queue());
        HodlrDevice<T>::sum_over(comm_,coeff,static_cast<int64_t>(k_)*nrhs,false,stream);
        apply.gemm(md,MagmaNoTrans,MagmaNoTrans,T(1),T(0),ctx.queue());
        if (op=='C' && fmm::gpu::is_complex_scalar<T>) conjugate(dy,dy,no,nrhs,stream);
        if (!device && ob) check_cuda(cudaMemcpyAsync(y,dy,ob,cudaMemcpyDeviceToHost,stream),"rectangular output");
        check_cuda(cudaStreamSynchronize(stream),"rectangular multiply");
        if (conjugated) heap.free(conjugated);
        // Host input storage is tracked separately because C changes dx.
        heap.free(coeff);
        if (!device) { heap.free(dy); heap.free(input_storage); }
        return (fmm::gpu::is_complex_scalar<T>?8.:2.)*(static_cast<double>(m_)+n_)*k_*nrhs;
    }
private:
    void conjugate(const T* a,T* out,int m,int n,cudaStream_t stream) {
        if (!m || !n) return;
        ItemList<ColScaleItem<T>> items;
        items.add({out,m,a,m,m,n,nullptr,1},m,n);
        meta_conj_.clear(); items.stage(meta_conj_);
        char* md=meta_conj_.upload(md_conj_,stream);
        launch_col_scale(items.device(md),1,m,n,stream);
    }
    void release() { if (store_) DeviceHeap::instance().free(store_); store_=nullptr; ready_=false; }
    int m_=0,n_=0,k_=0; MPI_Comm comm_=MPI_COMM_NULL;
    bool ready_=false;
    T *store_=nullptr,*u_=nullptr,*v_=nullptr;
    MetaBuilder meta_,meta_conj_; DeviceBuffer md_,md_conj_;
};
}} // namespace bpack::gpu
#endif
