#include <iostream>
#include <string>
#include <mkl.h>
#include <mkl_cluster_sparse_solver.h>
#include <mpi.h>
#include <Eigen/SparseCore>

using namespace std;
using myMatrix = Eigen::SparseMatrix<double,Eigen::RowMajor,MKL_INT>;

void set_a(int nz, int n, myMatrix& mat)
{
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;
  tripletList.reserve(nz);
  tripletList.push_back(T(0,0,1));
  tripletList.push_back(T(0,1,-1));
  tripletList.push_back(T(0,3,-3));
  tripletList.push_back(T(1,0,-2));
  tripletList.push_back(T(1,1,5));
  tripletList.push_back(T(2,2,4));
  tripletList.push_back(T(2,3,6));
  tripletList.push_back(T(2,4,4));
  tripletList.push_back(T(3,0,-3));
  tripletList.push_back(T(3,2,6));
  tripletList.push_back(T(3,3,7));
  tripletList.push_back(T(4,1,8));
  tripletList.push_back(T(4,4,-5));
  mat.setFromTriplets(tripletList.begin(), tripletList.end());
  mat.makeCompressed();
}

void set_b(double *b, int n)
{
  for(int i=0;i<n;i++) b[i]=1;
}

template <typename T>
void print_array(T *x, int n, string title)
{
  cout << "====" << endl;
  cout << title << endl;
  for(int i = 0; i < n; i++)
    {
      cout<<x[i]<<endl;
    }
  cout << "====" << endl;
}

int main(int argc, char **argv)
{
  void *pt[64] = { 0 };
  const MKL_INT maxfct = 1;
  const MKL_INT mnum = 1;
  const MKL_INT mtype = 11;
  MKL_INT phase = 11;
  const MKL_INT n = 5;
  int nz = 13;
  
  myMatrix A(n,n);

  double *a = NULL;
  MKL_INT ia[n+1];
  MKL_INT *ja = NULL;
  MKL_INT *perm = NULL;
  const MKL_INT nrhs = 1;
  MKL_INT iparm[64] = { 0 };
  iparm[0]=1; /* Solver default parameters overriden with provided by iparm */
  iparm[ 1] =  2; /* Use METIS for fill-in reordering */
  iparm[ 5] =  0; /* Write solution into x */
  iparm[ 7] =  2; /* Max number of iterative refinement steps */
  iparm[ 9] = 13; /* Perturb the pivot elements with 1E-13 */
  iparm[10] =  1; /* Use nonsymmetric permutation and scaling MPS */
  iparm[12] =  1; /* Switch on Maximum Weighted Matching algorithm (default for non-symmetric) */
  iparm[17] = -1; /* Output: Number of nonzeros in the factor LU */
  iparm[18] = -1; /* Output: Mflops for LU factorization */
  iparm[26] =  1; /* Check input data for correctness */
  iparm[34] =  1; //zero-based
  iparm[36] =  0; //CSR
  iparm[39] =  0; /* Input: matrix/rhs/solution stored on master */

  const MKL_INT msglvl = 1;
  double b[n];
  double x[n];
  MKL_INT error = 0;

  int size, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int comm =  MPI_Comm_c2f( MPI_COMM_WORLD );

  set_a(nz,n,A);
  a = A.valuePtr();
  for(int i = 0; i < n; i++) ia[i]=A.outerIndexPtr()[i];
  ia[n] = nz; 
  ja = A.innerIndexPtr();

  set_b(b, n);

  phase = 13;
  cluster_sparse_solver (pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, perm, &nrhs, iparm, &msglvl, b, x, &comm, &error);
  phase = 33;
  cluster_sparse_solver (pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, perm, &nrhs, iparm, &msglvl, b, x, &comm, &error);
  phase = -1;
  cluster_sparse_solver (pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, perm, &nrhs, iparm, &msglvl, b, x, &comm, &error);

  if(rank == 0)
    {
      print_array<double>(x,n,"x=");
    }
  
  MPI_Finalize();

}
