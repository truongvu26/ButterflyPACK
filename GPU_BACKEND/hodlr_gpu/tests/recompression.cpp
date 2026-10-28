// Compare native GPU QR/SVD/WY recompression against independent host BLAS
// products. Include mixed batches, QR/SVD thresholds, zero tau, and scaling.
#include "hodlr_gpu/hodlr_construct.hpp"
#include <mpi.h>
#include <cmath>
#include <complex>
#include <cstdio>
#include <numeric>
#include <random>

extern "C" {
void dgemm_(const char*,const char*,const int*,const int*,const int*,const double*,
            const double*,const int*,const double*,const int*,const double*,double*,const int*);
void zgemm_(const char*,const char*,const int*,const int*,const int*,const fmm::gpu::dcomplex*,
            const fmm::gpu::dcomplex*,const int*,const fmm::gpu::dcomplex*,const int*,
            const fmm::gpu::dcomplex*,fmm::gpu::dcomplex*,const int*);
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
template<typename T> std::vector<T> product(int m,int n,int k,const std::vector<T>& u,const std::vector<T>& v) {
    const T one(1.),zero(0.); const char N='N',trans='T'; std::vector<T> a(size_t(m)*n);
    if constexpr(std::is_same_v<T,double>) dgemm_(&N,&trans,&m,&n,&k,&one,u.data(),&m,v.data(),&n,&zero,a.data(),&m);
    else zgemm_(&N,&trans,&m,&n,&k,&one,u.data(),&m,v.data(),&n,&zero,a.data(),&m);
    return a;
}
template<typename T> struct Input { int m,n,k,kind; std::vector<T> u,v; };
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
    return out;
}
template<typename T> void run(const std::vector<Input<T>>& cases,bool batch,int rank,double tol) {
    using namespace bpack::gpu;
    for(size_t first=0;first<cases.size();first+=batch?cases.size():1) {
        const int count=batch?int(cases.size()):1;
        HodlrConstruct<T> c; DeviceSvd<T> svd;
        std::vector<int> blocks(count),ranks(count); std::iota(blocks.begin(),blocks.end(),0);
        for(int a=0;a<count;++a) {
            const auto& x=cases[first+a]; HodlrConstructTestAccess::inject(c,x.m,x.n,x.k,x.u,x.v);
        }
        c.recompress(count,blocks.data(),tol,1.e-300,svd,ranks.data());
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
            const double bound=std::max(2.e-8,2*std::sqrt(double(x.k))*tol);
            std::printf("recompression rank=%d scalar=%s batch=%d m=%d n=%d k=%d kind=%d retained=%d tol=%.2g error=%.12g bound=%.3g\n",
                rank,std::is_same_v<T,double>?"d":"z",batch,x.m,x.n,x.k,x.kind,ranks[a],tol,error,bound);std::fflush(stdout);
            if(!std::isfinite(error)||error>bound)throw std::runtime_error("GPU recompression changed the represented matrix");
        }
    }
}
int main(int argc,char** argv) {
    MPI_Init(&argc,&argv); int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    try {
        fmm::gpu::share_device(MPI_COMM_WORLD);
        auto d=inputs<double>();auto z=inputs<fmm::gpu::dcomplex>();
        for(double tol:{1.e-5,1.e-10}) {
            run(d,true,rank,tol);run(d,false,rank,tol);run(z,true,rank,tol);run(z,false,rank,tol);
        }
        std::printf("GPU recompression regression rank=%d PASS\n",rank);
    } catch(const std::exception& error) {
        std::fprintf(stderr,"rank=%d %s\n",rank,error.what());MPI_Abort(MPI_COMM_WORLD,1);
    }
    MPI_Finalize();
}
