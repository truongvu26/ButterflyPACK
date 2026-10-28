// Compare native GPU QR/SVD/WY recompression against independent host BLAS
// products. Include mixed batches, QR/SVD thresholds, zero tau, and scaling.
#include "hodlr_gpu/hodlr_construct.hpp"
#include <mpi.h>
#include <cmath>
#include <complex>
#include <cstdio>
#include <numeric>
#include <random>
#include <fstream>
#include <chrono>

extern "C" {
void dgemm_(const char*,const char*,const int*,const int*,const int*,const double*,
            const double*,const int*,const double*,const int*,const double*,double*,const int*);
void zgemm_(const char*,const char*,const int*,const int*,const int*,const fmm::gpu::dcomplex*,
            const fmm::gpu::dcomplex*,const int*,const fmm::gpu::dcomplex*,const int*,
            const fmm::gpu::dcomplex*,fmm::gpu::dcomplex*,const int*);
void dgeqrf_(const int*,const int*,double*,const int*,double*,double*,const int*,int*);
void zgeqrf_(const int*,const int*,fmm::gpu::dcomplex*,const int*,fmm::gpu::dcomplex*,
             fmm::gpu::dcomplex*,const int*,int*);
}

namespace bpack { namespace gpu {
struct HodlrConstructTestAccess {
    template<typename T> static void inject(HodlrConstruct<T>& c,int m,int n,int k,
                                            const std::vector<T>& u,const std::vector<T>& v) {
        c.blocks_.emplace_back(); auto& b=c.blocks_.back(); b.m=m; b.n=n;
        c.grow(b,k); b.rank=k;
        fmm::gpu::check_cuda(cudaMemcpy(b.u,u.data(),u.size()*sizeof(T),cudaMemcpyHostToDevice),"test U");
        fmm::gpu::check_cuda(cudaMemcpy(b.vt,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice),"test V");
    }
};
}}

// Independent LAPACK QR supplies realistic Gram/tau data, including exactly
// zero Householder coefficients. Padded leading dimensions exercise the
// batched setup kernel's indexing independently of the recompression driver.
template<typename T> struct WyInput {
    int k,nrhs,ldg,ldz;
    std::string kind;
    std::vector<T> gram,tau,rhs;
};
template<typename T> WyInput<T> wy_input(int k,int nrhs,const std::string& kind,bool padded=true) {
    const int m=k+7,ldg=k+(padded?3:0),ldz=k+(padded?5:0);
    std::mt19937_64 rng(98531+37*k+nrhs);std::uniform_real_distribution<double> rand(-1.,1.);
    auto value=[&]() -> T {
        const double re=rand(rng),im=rand(rng);
        if constexpr(std::is_same_v<T,double>)return re;
        else return T(re,im);
    };
    WyInput<T> out{k,nrhs,ldg,ldz,kind,std::vector<T>(size_t(ldg)*k,T(-9.)),
                   std::vector<T>(k),std::vector<T>(size_t(ldz)*nrhs,T(-7.))};
    std::vector<T> a(size_t(m)*k),y(size_t(m)*k);
    for(int j=0;j<k;++j)for(int i=0;i<m;++i) {
        T v=value()/std::sqrt(double(m));
        if(kind=="identity")v=T(i==j?1.:0.);
        else if(kind=="zero_columns" && j%3==0)v=T(0.);
        else if(kind=="ill_conditioned")v=v*std::pow(10.,-12.*j/std::max(1,k-1));
        else if(kind=="phase_diagonal") {
            if constexpr(std::is_same_v<T,double>)v=T(i==j?(j%2?-1.:1.):0.);
            else v=i==j?T(.3,.7):T(0.);
        }
        a[i+size_t(j)*m]=v;
    }
    int info=0,lwork=-1;T query;
    if constexpr(std::is_same_v<T,double>)dgeqrf_(&m,&k,a.data(),&m,out.tau.data(),&query,&lwork,&info);
    else zgeqrf_(&m,&k,a.data(),&m,out.tau.data(),&query,&lwork,&info);
    if(info)throw std::runtime_error("WY reference QR workspace query failed");
    if constexpr(std::is_same_v<T,double>)lwork=std::max(1,int(query));
    else lwork=std::max(1,int(query.re));
    std::vector<T> work(lwork);
    if constexpr(std::is_same_v<T,double>)dgeqrf_(&m,&k,a.data(),&m,out.tau.data(),work.data(),&lwork,&info);
    else zgeqrf_(&m,&k,a.data(),&m,out.tau.data(),work.data(),&lwork,&info);
    if(info)throw std::runtime_error("WY reference QR failed");
    for(int j=0;j<k;++j)for(int i=0;i<m;++i)
        y[i+size_t(j)*m]=i<j?T(0.):(i==j?T(1.):a[i+size_t(j)*m]);
    const T one(1.),zero(0.);const char N='N',H=std::is_same_v<T,double>?'T':'C';
    if constexpr(std::is_same_v<T,double>)
        dgemm_(&H,&N,&k,&k,&m,&one,y.data(),&m,y.data(),&m,&zero,out.gram.data(),&ldg);
    else zgemm_(&H,&N,&k,&k,&m,&one,y.data(),&m,y.data(),&m,&zero,out.gram.data(),&ldg);
    if(kind=="tiny_tau")for(int j=0;j<k;j+=2)out.tau[j]=T(1.e-300);
    for(int j=0;j<nrhs;++j)for(int i=0;i<k;++i)out.rhs[i+size_t(j)*ldz]=value();
    return out;
}

// Reference compact-WY larft recurrence, followed by ordinary multiplication.
// This forms T explicitly; the device implementation instead solves a
// row-scaled triangular system and must agree even when tau contains zeros.
template<typename T> std::vector<T> wy_reference(const WyInput<T>& x) {
    const int k=x.k;std::vector<T> t(size_t(k)*k),v(k),z=x.rhs;
    for(int i=0;i<k;++i) {
        if(!(x.tau[i]==T(0.))) {
            for(int j=0;j<i;++j)v[j]=T(0.)-x.tau[i]*x.gram[j+size_t(i)*x.ldg];
            for(int j=0;j<i;++j) {
                T acc(0.);
                for(int l=j;l<i;++l)acc+=t[j+size_t(l)*k]*v[l];
                t[j+size_t(i)*k]=acc;
            }
        }
        t[i+size_t(i)*k]=x.tau[i];
    }
    for(int c=0;c<x.nrhs;++c)for(int i=0;i<k;++i) {
        T acc(0.);
        for(int l=i;l<k;++l)acc+=t[i+size_t(l)*k]*x.rhs[l+size_t(c)*x.ldz];
        z[i+size_t(c)*x.ldz]=acc;
    }
    return z;
}

template<typename T> class WyDeviceBatch {
    using Clock=std::chrono::steady_clock;
    const std::vector<WyInput<T>>& input_;
    std::vector<void*> allocations_;
    std::vector<T*> gram_,tau_,rhs_,original_;
    bpack::gpu::ItemList<bpack::gpu::WySolveItem<T>> setup_;
    bpack::gpu::TrsmBatch<T> solve_;
    fmm::gpu::MetaBuilder meta_;
    fmm::gpu::DeviceBuffer metadata_;
    char* md_=nullptr;
    int max_k_=0,max_rhs_=0;
    T* allocate(size_t count) {
        T* p=bpack::gpu::DeviceHeap::instance().alloc<T>(count*sizeof(T));allocations_.push_back(p);return p;
    }
public:
    explicit WyDeviceBatch(const std::vector<WyInput<T>>& input):input_(input) {
        using namespace bpack::gpu;
        for(const auto& x:input_) {
            T* g=allocate(x.gram.size());T* t=allocate(x.tau.size());
            T* z=allocate(x.rhs.size());T* saved=allocate(x.rhs.size());T* matrix=allocate(size_t(x.k)*x.k);
            gram_.push_back(g);tau_.push_back(t);rhs_.push_back(z);original_.push_back(saved);
            fmm::gpu::check_cuda(cudaMemcpy(g,x.gram.data(),x.gram.size()*sizeof(T),cudaMemcpyHostToDevice),"WY test Gram");
            fmm::gpu::check_cuda(cudaMemcpy(t,x.tau.data(),x.tau.size()*sizeof(T),cudaMemcpyHostToDevice),"WY test tau");
            fmm::gpu::check_cuda(cudaMemcpy(saved,x.rhs.data(),x.rhs.size()*sizeof(T),cudaMemcpyHostToDevice),"WY test RHS");
            setup_.add({matrix,g,x.ldg,t,x.k,z,x.ldz,x.nrhs},x.k,x.nrhs);
            solve_.add(matrix,x.k,z,x.ldz,x.k,x.nrhs);
            max_k_=std::max(max_k_,x.k);max_rhs_=std::max(max_rhs_,x.nrhs);
        }
        setup_.stage(meta_);solve_.stage(meta_);
        md_=meta_.upload(metadata_,Context::instance().stream());
        fmm::gpu::check_cuda(cudaStreamSynchronize(Context::instance().stream()),"WY test metadata");
    }
    ~WyDeviceBatch() {
        cudaStreamSynchronize(bpack::gpu::Context::instance().stream());
        for(void* p:allocations_)bpack::gpu::DeviceHeap::instance().free(p);
    }
    void reset() {
        const auto stream=bpack::gpu::Context::instance().stream();
        for(size_t i=0;i<input_.size();++i)
            fmm::gpu::check_cuda(cudaMemcpyAsync(rhs_[i],original_[i],input_[i].rhs.size()*sizeof(T),
                                               cudaMemcpyDeviceToDevice,stream),"WY test reset");
        fmm::gpu::check_cuda(cudaStreamSynchronize(stream),"WY test reset fence");
    }
    double device() {
        auto& context=bpack::gpu::Context::instance();const auto start=Clock::now();
        bpack::gpu::launch_wy_solve_setup(setup_.device(md_),setup_.count(),max_k_,max_rhs_,context.stream());
        solve_.solve(md_,MagmaUpper,MagmaNonUnit,context.queue());
        fmm::gpu::check_cuda(cudaStreamSynchronize(context.stream()),"WY device solve");
        return std::chrono::duration<double>(Clock::now()-start).count();
    }
    double legacy_host() {
        const auto start=Clock::now();
        for(size_t i=0;i<input_.size();++i) {
            auto x=input_[i];
            fmm::gpu::check_cuda(cudaMemcpy(x.gram.data(),gram_[i],x.gram.size()*sizeof(T),cudaMemcpyDeviceToHost),"legacy WY Gram");
            fmm::gpu::check_cuda(cudaMemcpy(x.tau.data(),tau_[i],x.tau.size()*sizeof(T),cudaMemcpyDeviceToHost),"legacy WY tau");
            fmm::gpu::check_cuda(cudaMemcpy(x.rhs.data(),rhs_[i],x.rhs.size()*sizeof(T),cudaMemcpyDeviceToHost),"legacy WY RHS");
            const auto z=wy_reference(x);
            fmm::gpu::check_cuda(cudaMemcpy(rhs_[i],z.data(),z.size()*sizeof(T),cudaMemcpyHostToDevice),"legacy WY result");
        }
        return std::chrono::duration<double>(Clock::now()-start).count();
    }
    void verify(int rank) {
        for(size_t a=0;a<input_.size();++a) {
            const auto& x=input_[a];const auto expected=wy_reference(x);std::vector<T> actual(x.rhs.size());
            fmm::gpu::check_cuda(cudaMemcpy(actual.data(),rhs_[a],actual.size()*sizeof(T),cudaMemcpyDeviceToHost),"WY test result");
            long double diff=0.,norm=0.;int zeros=0;
            for(const auto& tau:x.tau)if(tau==T(0.))++zeros;
            for(int j=0;j<x.nrhs;++j)for(int i=0;i<x.ldz;++i) {
                const size_t p=i+size_t(j)*x.ldz;
                if(i>=x.k) {if(!(actual[p]==x.rhs[p]))throw std::runtime_error("WY setup overwrote RHS padding");continue;}
                diff+=fmm::gpu::abs2(actual[p]-expected[p]);norm+=fmm::gpu::abs2(expected[p]);
            }
            const double error=norm?std::sqrt(diff/norm):std::sqrt(diff);
            const double bound=2.e-12*std::max(1.,std::sqrt(double(x.k)));
            std::printf("direct_wy rank=%d scalar=%s k=%d nrhs=%d kind=%s zero_tau=%d error=%.12g bound=%.3g\n",
                rank,std::is_same_v<T,double>?"d":"z",x.k,x.nrhs,x.kind.c_str(),zeros,error,bound);
            if(!std::isfinite(error)||error>bound)throw std::runtime_error("GPU WY solve disagrees with host larft recurrence");
        }
        std::fflush(stdout);
    }
};

template<typename T> void check_wy(int rank) {
    std::vector<WyInput<T>> cases;
    for(int k:{1,2,7,31,33,127,257})for(const char* kind:{"random","zero_columns","identity","ill_conditioned","phase_diagonal","tiny_tau"})
        cases.push_back(wy_input<T>(k,1+k%7,kind));
    WyDeviceBatch<T> batch(cases);batch.reset();batch.device();batch.verify(rank);
}
template<typename T> void benchmark_wy(int rank) {
    // Isolated zero-tau fallback benchmark, not whole recompression timing.
    // Resident Gram/tau/RHS and fixed shared metadata are created before timing.
    // Reset copies are excluded for both methods. Legacy timing includes all
    // required D2H/H2D copies and the host recurrence/application. Device
    // timing includes setup, TRSM and the completion fence, not metadata upload.
    for(int k:{256,512,1024}) {
        std::vector<WyInput<T>> cases;
        cases.push_back(wy_input<T>(k,k/2,"zero_columns",false));
        cases.push_back(wy_input<T>(k,k/2,"ill_conditioned",false));
        // A zero coefficient makes both fixtures eligible for the former
        // host fallback. The remaining ill-conditioned QR data is retained.
        cases.back().tau[k/2]=T(0.);cases.back().kind="ill_conditioned_zero_tau";
        WyDeviceBatch<T> batch(cases);
        batch.reset();batch.device();batch.verify(rank);
        batch.reset();batch.legacy_host();batch.verify(rank);
        std::vector<double> device,host;
        for(int trial=0;trial<5;++trial) {
            double d,h;
            if(trial%2) {
                batch.reset();h=batch.legacy_host();
                batch.reset();d=batch.device();
            } else {
                batch.reset();d=batch.device();
                batch.reset();h=batch.legacy_host();
            }
            if(trial) {device.push_back(d);host.push_back(h);}
        }
        std::sort(device.begin(),device.end());std::sort(host.begin(),host.end());
        const double d=(device[1]+device[2])/2,h=(host[1]+host[2])/2;
        std::printf("wy_benchmark rank=%d scalar=%s k=%d nrhs=%d batch=2 repetitions=4 scope=isolated_zero_tau_fallback metadata_upload=excluded input_creation=excluded reset=excluded legacy_transfers=included device_median_seconds=%.9g legacy_host_median_seconds=%.9g speedup=%.6g\n",
            rank,std::is_same_v<T,double>?"d":"z",k,k/2,d,h,h/d);std::fflush(stdout);
    }
}
template<typename T> std::vector<T> product(int m,int n,int k,const std::vector<T>& u,const std::vector<T>& v) {
    const T one(1.),zero(0.); const char N='N',trans='T'; std::vector<T> a(size_t(m)*n);
    if constexpr(std::is_same_v<T,double>) dgemm_(&N,&trans,&m,&n,&k,&one,u.data(),&m,v.data(),&n,&zero,a.data(),&m);
    else zgemm_(&N,&trans,&m,&n,&k,&one,u.data(),&m,v.data(),&n,&zero,a.data(),&m);
    return a;
}
template<typename T> struct Input { int m,n,k,kind; std::vector<T> u,v; };
template<typename T> void check_svd(const Input<T>& x,int rank) {
    auto expected=product(x.m,x.n,x.k,x.u,x.v);
    using namespace bpack::gpu;
    auto& heap=DeviceHeap::instance(); const int k=std::min(x.m,x.n);
    T* a=heap.alloc<T>(expected.size()*sizeof(T));
    T* u=heap.alloc<T>(size_t(x.m)*k*sizeof(T));T* v=heap.alloc<T>(size_t(x.n)*k*sizeof(T));
    double* s=heap.alloc<double>(size_t(k)*sizeof(double));
    fmm::gpu::check_cuda(cudaMemcpy(a,expected.data(),expected.size()*sizeof(T),cudaMemcpyHostToDevice),"direct SVD input");
    DeviceSvd<T> svd;double perturbation=svd.svd_device(x.m,x.n,a,s,u,v);
    std::vector<T> hu(size_t(x.m)*k),hv(size_t(x.n)*k);std::vector<double> hs(k);
    fmm::gpu::check_cuda(cudaMemcpy(hu.data(),u,hu.size()*sizeof(T),cudaMemcpyDeviceToHost),"direct SVD U");
    fmm::gpu::check_cuda(cudaMemcpy(hv.data(),v,hv.size()*sizeof(T),cudaMemcpyDeviceToHost),"direct SVD V");
    fmm::gpu::check_cuda(cudaMemcpy(hs.data(),s,hs.size()*sizeof(double),cudaMemcpyDeviceToHost),"direct SVD singular values");
    for(int j=0;j<k;++j) {
        for(int i=0;i<x.m;++i)hu[i+size_t(j)*x.m]=hu[i+size_t(j)*x.m]*T(hs[j]);
        if constexpr(!std::is_same_v<T,double>)for(int i=0;i<x.n;++i)hv[i+size_t(j)*x.n].im*=-1;
    }
    auto actual=product(x.m,x.n,k,hu,hv);long double error=0,norm=0;
    for(size_t i=0;i<actual.size();++i){error+=fmm::gpu::abs2(actual[i]-expected[i]);norm+=fmm::gpu::abs2(expected[i]);}
    double relative=std::sqrt(error/norm);
    std::printf("direct_svd rank=%d m=%d n=%d k=%d err_sigma=%.12g norm=%.12g relative_error=%.12g\n",
                rank,x.m,x.n,k,perturbation,double(std::sqrt(norm)),relative);std::fflush(stdout);
    for(void* p:{static_cast<void*>(s),static_cast<void*>(v),static_cast<void*>(u),static_cast<void*>(a)})heap.free(p);
    if(!std::isfinite(relative)||relative>1.e-7)throw std::runtime_error("Direct GPU SVD failed independent reconstruction");
}
template<typename T> std::vector<Input<T>> inputs() {
    std::mt19937_64 rng(98531); std::uniform_real_distribution<double> rand(-1.,1.);
    auto value=[&]() -> T {
        double re=rand(rng),im=rand(rng);
        if constexpr(std::is_same_v<T,double>) return re;
        else return T(re,im);
    };
    std::vector<Input<T>> out;
    const int ranks[]={32,64,255,256,511,512,640};
    for(int a=0;a<7;++a) {
        int k=ranks[a],kind=a%3,m=kind==1?k:k+13,n=k+31;
        Input<T> x{m,n,k,kind,std::vector<T>(size_t(m)*k),std::vector<T>(size_t(n)*k)};
        for(int j=0;j<k;++j) {
            double scale=kind==2?std::pow(10.,-10.*j/(k-1)):1.;
            for(int i=0;i<m;++i)x.u[i+size_t(j)*m]=value()*scale/std::sqrt(double(m));
            for(int i=0;i<n;++i)x.v[i+size_t(j)*n]=value()/std::sqrt(double(n));
        }
        if(kind==1) for(int j:{0,k/2,k-1})
            std::fill(x.u.begin()+size_t(j)*m,x.u.begin()+size_t(j+1)*m,T(0.));
        out.push_back(std::move(x));
    }
    // Exact-factor fallback uses I*A or A*I. These give zero Householder
    // coefficients with nonzero R diagonals, on both sides of the product.
    for(int k:{200,512})for(int side:{0,1}) {
        const int m=side?k+31:k,n=side?k:k+31;
        Input<T> x{m,n,k,3+side,std::vector<T>(size_t(m)*k),std::vector<T>(size_t(n)*k)};
        for(int j=0;j<k;++j) {
            for(int i=0;i<m;++i)x.u[i+size_t(j)*m]=side?value():T(i==j?1.:0.);
            for(int i=0;i<n;++i)x.v[i+size_t(j)*n]=side?T(i==j?1.:0.):value();
        }
        out.push_back(std::move(x));
    }
    // A square, exactly rank-deficient core is a realistic frontal SVD
    // input. Random full-rank factors do not exercise its null space.
    {
        const int k=576;
        Input<T> x{k,k,k,6,std::vector<T>(size_t(k)*k),std::vector<T>(size_t(k)*k)};
        for(int j=0;j<k;++j) {
            x.u[j+size_t(j)*k]=T(1.);
            if(j<128) {
                if constexpr(std::is_same_v<T,double>)x.v[j+size_t(j)*k]=.1/(1.+.01*j);
                else x.v[j+size_t(j)*k]=T(.1/(1.+.01*j),.01);
            }
        }
        out.push_back(std::move(x));
    }
    return out;
}
template<typename T> void run(const std::vector<Input<T>>& cases,bool batch,int rank,double tol,bool adopt=false) {
    using namespace bpack::gpu;
    for(size_t first=0;first<cases.size();first+=batch?cases.size():1) {
        const int count=batch?int(cases.size()):1;
        HodlrConstruct<T> c; DeviceSvd<T> svd;
        std::vector<int> blocks(count),ranks(count); std::iota(blocks.begin(),blocks.end(),0);
        for(int a=0;a<count;++a) {
            const auto& x=cases[first+a]; HodlrConstructTestAccess::inject(c,x.m,x.n,x.k,x.u,x.v);
        }
        c.recompress(count,blocks.data(),tol,1.e-300,svd,ranks.data());
        if(adopt) {
            c.adopt_output(count,blocks.data(),ranks.data());
            c.recompress(count,blocks.data(),tol,1.e-300,svd,ranks.data());
        }
        std::vector<std::vector<T>> u(count),v(count); std::vector<T*> up(count),vp(count);
        for(int a=0;a<count;++a) {
            const auto& x=cases[first+a]; u[a].resize(size_t(x.m)*ranks[a]);v[a].resize(size_t(x.n)*ranks[a]);
            up[a]=u[a].data();vp[a]=v[a].data();
        }
        c.download(count,blocks.data(),ranks.data(),up.data(),vp.data());
        for(int a=0;a<count;++a) {
            const auto& x=cases[first+a]; auto expected=product(x.m,x.n,x.k,x.u,x.v);
            auto actual=product(x.m,x.n,ranks[a],u[a],v[a]); long double diff=0,norm=0;
            for(size_t i=0;i<actual.size();++i){diff+=fmm::gpu::abs2(actual[i]-expected[i]);norm+=fmm::gpu::abs2(expected[i]);}
            const double error=std::sqrt(diff/norm);
            const double bound=std::max(2.e-8,(adopt?4:2)*std::sqrt(double(x.k))*tol);
            std::printf("recompression rank=%d scalar=%s batch=%d adopt=%d m=%d n=%d k=%d kind=%d retained=%d tol=%.2g error=%.12g bound=%.3g\n",
                rank,std::is_same_v<T,double>?"d":"z",batch,adopt,x.m,x.n,x.k,x.kind,ranks[a],tol,error,bound);std::fflush(stdout);
            if(!std::isfinite(error)||error>bound)throw std::runtime_error("GPU recompression changed the represented matrix");
        }
    }
}
int main(int argc,char** argv) {
    MPI_Init(&argc,&argv); int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    try {
        fmm::gpu::share_device(MPI_COMM_WORLD);
        const bool wy_only=argc==2 && std::string(argv[1])=="--wy-only";
        const bool wy_benchmark=argc==2 && std::string(argv[1])=="--wy-benchmark";
        if(argc==1 || wy_only || wy_benchmark) {
            check_wy<double>(rank);check_wy<fmm::gpu::dcomplex>(rank);
        }
        if(wy_only || wy_benchmark) {
            if(wy_benchmark) {benchmark_wy<double>(rank);benchmark_wy<fmm::gpu::dcomplex>(rank);}
            std::printf("GPU WY regression rank=%d PASS\n",rank);std::fflush(stdout);
            MPI_Finalize();return 0;
        }
        auto d=inputs<double>();auto z=inputs<fmm::gpu::dcomplex>();
        const bool direct=argc>1 && std::string(argv[1])=="--svd";
        if(argc>1) {
            d.clear();z.clear();
            if(direct && argc<3)throw std::runtime_error("--svd needs a factor capture");
            for(int f=direct?2:1;f<argc;++f) {
                std::ifstream file(argv[f],std::ios::binary);uint64_t h[5];file.read(reinterpret_cast<char*>(h),sizeof(h));
                if(!file || h[0]!=0x524f544341465042ULL || !h[1] || !h[2] || !h[3] ||
                   h[1]>100000 || h[2]>100000 || h[3]>std::min(h[1],h[2]) ||
                   (h[4]!=8 && h[4]!=16) || (h[1]+h[2])*h[3]*h[4]>(uint64_t{64}<<20))
                    throw std::runtime_error("Invalid GPU factor capture");
                auto read=[&](auto& cases) {
                    using T=typename std::decay_t<decltype(cases)>::value_type;
                    T x{int(h[1]),int(h[2]),int(h[3]),5,{},{}};
                    x.u.resize(h[1]*h[3]);x.v.resize(h[2]*h[3]);
                    file.read(reinterpret_cast<char*>(x.u.data()),x.u.size()*h[4]);
                    file.read(reinterpret_cast<char*>(x.v.data()),x.v.size()*h[4]);
                    if(!file)throw std::runtime_error("Truncated GPU factor capture");
                    cases.push_back(std::move(x));
                };
                if(h[4]==8)read(d);else read(z);
            }
        }
        if(direct) {for(auto& x:d)check_svd(x,rank);for(auto& x:z)check_svd(x,rank);}
        else {
            if(argc==1) {
                for(auto& x:d)if(x.kind==6 || (x.k==200 && x.kind>=3))check_svd(x,rank);
                for(auto& x:z)if(x.kind==6 || (x.k==200 && x.kind>=3))check_svd(x,rank);
            }
            for(double tol:{1.e-5,1.e-10}) {
                if(!d.empty()) {run(d,true,rank,tol);run(d,false,rank,tol);run(d,true,rank,tol,true);}
                if(!z.empty()) {run(z,true,rank,tol);run(z,false,rank,tol);run(z,true,rank,tol,true);}
            }
        }
        std::printf("GPU recompression regression rank=%d PASS\n",rank);
    } catch(const std::exception& error) {
        std::fprintf(stderr,"rank=%d %s\n",rank,error.what());MPI_Abort(MPI_COMM_WORLD,1);
    }
    MPI_Finalize();
}
