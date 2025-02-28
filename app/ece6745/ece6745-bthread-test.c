//========================================================================
// Unit tests for bthreads
//========================================================================

#include "ece6745-misc.h"
#include "ece6745-check.h"
#include "ece6745-wprint.h"
#include "ece6745-bthread.h"

//------------------------------------------------------------------------
// Test num cores and core id
//------------------------------------------------------------------------

void test_case_1_numcores_coreid()
{
  ECE6745_CHECK( L"test_case_1_numcores_coreid" );

  int ncores = ece6745_bthread_get_num_cores();
  ECE6745_CHECK_TRUE( ncores == 1 || ncores == 4 );
  ECE6745_CHECK_INT_EQ( ece6745_bthread_get_core_id(), 0 );
}

//------------------------------------------------------------------------
// Test spawning work on each core and all cores
//------------------------------------------------------------------------

void test_func0( void* args_vptr )
{
  int* result_ptr = (int*) args_vptr;
  *result_ptr = 1;
}

void test_case_2_spawn_core1()
{
  ECE6745_CHECK( L"test_case_2_spawn_core1" );

  int result = 0;

  ece6745_bthread_spawn( 1, &test_func0, &result );
  ece6745_bthread_join(1);

  ECE6745_CHECK_INT_EQ( result, 1 );
}

void test_case_3_spawn_core2()
{
  ECE6745_CHECK( L"test_case_3_spawn_core2" );

  int result = 0;

  ece6745_bthread_spawn( 2, &test_func0, &result );
  ece6745_bthread_join(2);

  ECE6745_CHECK_INT_EQ( result, 1 );
}

void test_case_4_spawn_core3()
{
  ECE6745_CHECK( L"test_case_4_spawn_core3" );

  int result = 0;

  ece6745_bthread_spawn( 3, &test_func0, &result );
  ece6745_bthread_join(3);

  ECE6745_CHECK_INT_EQ( result, 1 );
}

void test_case_5_spawn_all_cores()
{
  ECE6745_CHECK( L"test_case_5_spawn_all_cores" );

  int result1 = 0;
  int result2 = 0;
  int result3 = 0;

  ece6745_bthread_spawn( 1, &test_func0, &result1 );
  ece6745_bthread_spawn( 2, &test_func0, &result2 );
  ece6745_bthread_spawn( 3, &test_func0, &result3 );

  ece6745_bthread_join(1);
  ece6745_bthread_join(2);
  ece6745_bthread_join(3);

  ECE6745_CHECK_INT_EQ( result1, 1 );
  ECE6745_CHECK_INT_EQ( result2, 1 );
  ECE6745_CHECK_INT_EQ( result3, 1 );
}

//------------------------------------------------------------------------
// Test spawning work on each core and all cores
//------------------------------------------------------------------------

void test_func1( void* args_vptr )
{
  int* result_ptr = (int*) args_vptr;
  *result_ptr = 2;
}

void test_case_6_multi_spawn_core1()
{
  ECE6745_CHECK( L"test_case_6_multi_spawn_core1" );

  int result = 0;

  ece6745_bthread_spawn( 1, &test_func0, &result );
  ece6745_bthread_join(1);

  ECE6745_CHECK_INT_EQ( result, 1 );

  ece6745_bthread_spawn( 1, &test_func1, &result );
  ece6745_bthread_join(1);

  ECE6745_CHECK_INT_EQ( result, 2 );
}

void test_case_7_multi_spawn_core2()
{
  ECE6745_CHECK( L"test_case_7_multi_spawn_core2" );

  int result = 0;

  ece6745_bthread_spawn( 2, &test_func0, &result );
  ece6745_bthread_join(2);

  ECE6745_CHECK_INT_EQ( result, 1 );

  ece6745_bthread_spawn( 2, &test_func1, &result );
  ece6745_bthread_join(2);

  ECE6745_CHECK_INT_EQ( result, 2 );
}

void test_case_8_multi_spawn_core3()
{
  ECE6745_CHECK( L"test_case_8_multi_spawn_core3" );

  int result = 0;

  ece6745_bthread_spawn( 3, &test_func0, &result );
  ece6745_bthread_join(3);

  ECE6745_CHECK_INT_EQ( result, 1 );

  ece6745_bthread_spawn( 3, &test_func1, &result );
  ece6745_bthread_join(3);

  ECE6745_CHECK_INT_EQ( result, 2 );
}

void test_case_9_multi_spawn_all_cores()
{
  ECE6745_CHECK( L"test_case_9_multi_spawn_all_cores" );

  int result1 = 0;
  int result2 = 0;
  int result3 = 0;

  ece6745_bthread_spawn( 1, &test_func0, &result1 );
  ece6745_bthread_spawn( 2, &test_func0, &result2 );
  ece6745_bthread_spawn( 3, &test_func0, &result3 );

  ece6745_bthread_join(1);
  ece6745_bthread_join(2);
  ece6745_bthread_join(3);

  ECE6745_CHECK_INT_EQ( result1, 1 );
  ECE6745_CHECK_INT_EQ( result2, 1 );
  ECE6745_CHECK_INT_EQ( result3, 1 );

  ece6745_bthread_spawn( 1, &test_func1, &result1 );
  ece6745_bthread_spawn( 2, &test_func1, &result2 );
  ece6745_bthread_spawn( 3, &test_func1, &result3 );

  ece6745_bthread_join(1);
  ece6745_bthread_join(2);
  ece6745_bthread_join(3);

  ECE6745_CHECK_INT_EQ( result1, 2 );
  ECE6745_CHECK_INT_EQ( result2, 2 );
  ECE6745_CHECK_INT_EQ( result3, 2 );
}

//------------------------------------------------------------------------
// test_case_10_spawn_args()
//------------------------------------------------------------------------

typedef struct
{
  int* result_ptr;
  int  result_value;
}
test_func2_arg_t;

void test_func2( void* arg_vptr )
{
  test_func2_arg_t* arg_ptr = (test_func2_arg_t*) arg_vptr;

  int* result_ptr   = arg_ptr->result_ptr;
  int  result_value = arg_ptr->result_value;

  *result_ptr = result_value;
}

void test_case_10_spawn_args()
{
  ECE6745_CHECK( L"test_case_10_spawn_args" );

  int result1 = 0;
  int result2 = 0;
  int result3 = 0;

  test_func2_arg_t arg1 = { &result1, 1 };
  test_func2_arg_t arg2 = { &result2, 2 };
  test_func2_arg_t arg3 = { &result3, 3 };

  ece6745_bthread_spawn( 1, &test_func2, &arg1 );
  ece6745_bthread_spawn( 2, &test_func2, &arg2 );
  ece6745_bthread_spawn( 3, &test_func2, &arg3 );

  ece6745_bthread_join(1);
  ece6745_bthread_join(2);
  ece6745_bthread_join(3);

  ECE6745_CHECK_INT_EQ( result1, 1 );
  ECE6745_CHECK_INT_EQ( result2, 2 );
  ECE6745_CHECK_INT_EQ( result3, 3 );
}

//------------------------------------------------------------------------
// test_case_11_spawn_accumulate()
//------------------------------------------------------------------------

typedef struct
{
  int* src;
  int* sum_ptr;
  int  first;
  int  last;
}
test_func3_arg_t;

void test_func3( void* arg_vptr )
{
  test_func3_arg_t* arg_ptr = (test_func3_arg_t*) arg_vptr;

  int* src     = arg_ptr->src;
  int* sum_ptr = arg_ptr->sum_ptr;
  int  first   = arg_ptr->first;
  int  last    = arg_ptr->last;

  int sum = 0;
  for ( int i = first; i < last; i++ )
    sum += src[i];

  *sum_ptr = sum;
}

void test_case_11_spawn_accumulate()
{
  ECE6745_CHECK( L"test_case_11_spawn_accumulate" );

  int src[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
  int size  = 16;

  int block_size = size/4;

  // Create four argument structures that include the array pointers and
  // what elements each core should process.

  int sum0 = 0;
  int sum1 = 0;
  int sum2 = 0;
  int sum3 = 0;

  test_func3_arg_t arg0 = { src, &sum0, 0*block_size, 1*block_size };
  test_func3_arg_t arg1 = { src, &sum1, 1*block_size, 2*block_size };
  test_func3_arg_t arg2 = { src, &sum2, 2*block_size, 3*block_size };
  test_func3_arg_t arg3 = { src, &sum3, 3*block_size, size         };

  // Spawn work onto cores 1, 2, and 3.

  ece6745_bthread_spawn( 1, &test_func3, &arg1 );
  ece6745_bthread_spawn( 2, &test_func3, &arg2 );
  ece6745_bthread_spawn( 3, &test_func3, &arg3 );

  // Have core 0 also do some work.

  test_func3( &arg0 );

  // Wait for core 1, 2, and 3 to finish.

  ece6745_bthread_join(1);
  ece6745_bthread_join(2);
  ece6745_bthread_join(3);

  ECE6745_CHECK_INT_EQ( sum0, 10 );
  ECE6745_CHECK_INT_EQ( sum1, 26 );
  ECE6745_CHECK_INT_EQ( sum2, 42 );
  ECE6745_CHECK_INT_EQ( sum3, 58 );
}

//------------------------------------------------------------------------
// main
//------------------------------------------------------------------------

int main( int argc, char** argv )
{
  __n = ( argc == 1 ) ? 0 : ece6745_atoi( argv[1] );

  if ( (__n <= 0) || (__n ==  1) ) test_case_1_numcores_coreid();
  if ( (__n <= 0) || (__n ==  2) ) test_case_2_spawn_core1();
  if ( (__n <= 0) || (__n ==  3) ) test_case_3_spawn_core2();
  if ( (__n <= 0) || (__n ==  4) ) test_case_4_spawn_core3();
  if ( (__n <= 0) || (__n ==  5) ) test_case_5_spawn_all_cores();
  if ( (__n <= 0) || (__n ==  6) ) test_case_6_multi_spawn_core1();
  if ( (__n <= 0) || (__n ==  7) ) test_case_7_multi_spawn_core2();
  if ( (__n <= 0) || (__n ==  8) ) test_case_8_multi_spawn_core3();
  if ( (__n <= 0) || (__n ==  9) ) test_case_9_multi_spawn_all_cores();
  if ( (__n <= 0) || (__n == 10) ) test_case_10_spawn_args();
  if ( (__n <= 0) || (__n == 11) ) test_case_11_spawn_accumulate();

  ece6745_wprintf( L"\n\n" );
  return ece6745_check_status;
}

