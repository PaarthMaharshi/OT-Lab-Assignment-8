/* Track D supplement (PPT 2): data locality, false sharing, synchronisation, race conditions.
   Measured on this machine with OpenMP. Build: gcc -O2 -fopenmp locality_sync.c -o locality_sync
   Output: one line per measurement, "key value". */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 4096                     /* 4096 x 4096 doubles = 128 MB, far larger than the caches */
#define M 768                      /* matrix multiply size */
#define BS 32                      /* tile: 3 * 32 * 32 doubles = 24 KB fits a 32 KB L1 (slide 18) */

static double now(void) { return omp_get_wtime(); }

static double best_of(double (*f)(void), int reps) {
  double b = 1e30;
  for (int r = 0; r < reps; r++) { double t = f(); if (t < b) b = t; }
  return b;
}

static double *A, *B, *C;
static double sink;

/* ---------- loop order: row-wise (stride 1) vs column-wise (stride N) sums ---------- */
static double row_sum(void) {
  double t = now(), s = 0;
  for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) s += A[(size_t)i * N + j];
  sink += s; return now() - t;
}
static double col_sum(void) {
  double t = now(), s = 0;
  for (int j = 0; j < N; j++) for (int i = 0; i < N; i++) s += A[(size_t)i * N + j];
  sink += s; return now() - t;
}

/* ---------- matrix multiply: naive ijk vs cache-blocked ---------- */
static double mm_naive(void) {
  double t = now();
  for (int i = 0; i < M; i++) for (int j = 0; j < M; j++) {
    double s = 0;
    for (int k = 0; k < M; k++) s += A[i * M + k] * B[k * M + j];
    C[i * M + j] = s;
  }
  sink += C[7]; return now() - t;
}
static double mm_blocked(void) {
  double t = now();
  memset(C, 0, sizeof(double) * M * M);
  for (int ii = 0; ii < M; ii += BS) for (int kk = 0; kk < M; kk += BS) for (int jj = 0; jj < M; jj += BS)
    for (int i = ii; i < ii + BS; i++) for (int k = kk; k < kk + BS; k++) {
      double a = A[i * M + k];
      for (int j = jj; j < jj + BS; j++) C[i * M + j] += a * B[k * M + j];
    }
  sink += C[7]; return now() - t;
}

/* ---------- false sharing: per-thread counters in one cache line vs padded ---------- */
#define ITERS 50000000L
static volatile long packed[64];
static struct { volatile long v; char pad[56]; } padded[64];
static double fs_packed(void) {
  double t = now();
#pragma omp parallel num_threads(4)
  { int id = omp_get_thread_num(); for (long i = 0; i < ITERS; i++) packed[id]++; }
  return now() - t;
}
static double fs_padded(void) {
  double t = now();
#pragma omp parallel num_threads(4)
  { int id = omp_get_thread_num(); for (long i = 0; i < ITERS; i++) padded[id].v++; }
  return now() - t;
}

/* ---------- synchronisation: summing 1e8 numbers with 4 threads ---------- */
#define S 100000000L
static double sum_critical(void) {
  double t = now(), s = 0;
#pragma omp parallel for num_threads(4)
  for (long i = 0; i < S / 10; i++) {        /* 1e7 adds: a critical section per add is very slow */
#pragma omp critical
    s += 1.0;
  }
  sink += s; return now() - t;
}
static double sum_atomic(void) {
  double t = now(), s = 0;
#pragma omp parallel for num_threads(4)
  for (long i = 0; i < S / 10; i++) {
#pragma omp atomic
    s += 1.0;
  }
  sink += s; return now() - t;
}
static double sum_reduction(void) {
  double t = now(), s = 0;
#pragma omp parallel for num_threads(4) reduction(+:s)
  for (long i = 0; i < S / 10; i++) s += 1.0;
  sink += s; return now() - t;
}
static double sum_serial(void) {
  double t = now(), s = 0;
  for (long i = 0; i < S / 10; i++) s += 1.0;
  sink += s; return now() - t;
}

int main(void) {
  A = malloc(sizeof(double) * (size_t)N * N);
  B = malloc(sizeof(double) * M * M);
  C = malloc(sizeof(double) * M * M);
  for (size_t i = 0; i < (size_t)N * N; i++) A[i] = (double)(i % 7);
  for (int i = 0; i < M * M; i++) B[i] = (double)(i % 5);
  printf("row_sum_s %.4f\n", best_of(row_sum, 3));
  printf("col_sum_s %.4f\n", best_of(col_sum, 3));
  for (int i = 0; i < M * M; i++) A[i] = (double)(i % 3);
  double t1 = best_of(mm_naive, 2);
  double c_naive = 0; for (int i = 0; i < M * M; i++) c_naive += C[i];
  double t2 = best_of(mm_blocked, 2);
  double c_block = 0; for (int i = 0; i < M * M; i++) c_block += C[i];
  printf("matmul_naive_s %.4f\nmatmul_blocked_s %.4f\nmatmul_results_equal %d\n", t1, t2, c_naive == c_block);
  printf("false_sharing_packed_s %.4f\n", best_of(fs_packed, 3));
  printf("false_sharing_padded_s %.4f\n", best_of(fs_padded, 3));
  printf("sum_serial_s %.4f\n", best_of(sum_serial, 3));
  printf("sum_critical_s %.4f\n", best_of(sum_critical, 1));
  printf("sum_atomic_s %.4f\n", best_of(sum_atomic, 3));
  printf("sum_reduction_s %.4f\n", best_of(sum_reduction, 3));
  /* race condition: unsynchronised increments lose updates */
  long lost_runs = 0, expected = 4L * 1000000L, worst = expected;
  for (int r = 0; r < 5; r++) {
    volatile long x = 0;
#pragma omp parallel num_threads(4)
    for (long i = 0; i < 1000000L; i++) x = x + 1;   /* read-modify-write, no lock */
    if (x != expected) lost_runs++;
    if (x < worst) worst = x;
  }
  printf("race_expected %ld\nrace_worst_result %ld\nrace_runs_with_lost_updates_of_5 %ld\n", expected, worst, lost_runs);
  printf("sink %g\n", sink > 0 ? 1.0 : 0.0);
  return 0;
}
