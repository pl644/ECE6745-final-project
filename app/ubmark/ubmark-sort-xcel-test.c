//========================================================================
// Unit tests for ubmark sort
//========================================================================

#include "ece6745.h"
#include "ubmark-sort-xcel.h"
#include "ubmark-sort.dat"

//------------------------------------------------------------------------
// is_sorted
//------------------------------------------------------------------------
// Helper function that returns 1 if sorted and 0 if unsorted

int is_sorted( int* x, int n )
{
  for ( int i = 0; i < n-1; i++ ) {
    if ( x[i] > x[i+1] )
      return 0;
  }
  return 1;
}

//------------------------------------------------------------------------
// test_case_1_sort_basic
//------------------------------------------------------------------------

void test_case_1_sort_basic()
{
  ECE6745_CHECK( L"test_case_1_sort_basic" );

  int a[]     = { 4, 3, 6, 5, };
  int a_ref[] = { 3, 4, 5, 6, };

  ubmark_sort_xcel( a, 4 );

  for ( int i = 0; i < 4; i++ )
    ECE6745_CHECK_INT_EQ( a[i] , a_ref[i] );

  ECE6745_CHECK_INT_EQ( ece6745_get_heap_usage(), 0 );
}

//------------------------------------------------------------------------
// test_case_2_sort_five_elements
//------------------------------------------------------------------------
// A simple test case that tests sorting 5 numbers

void test_case_2_sort_five_element()
{
  ECE6745_CHECK( L"test_case_2_sort_five_elements" );

  int a[]     = { 4, 3, 5, 6, 10 };
  int a_ref[] = { 3, 4, 5, 6, 10 };

  ubmark_sort_xcel( a, 5 );

  for ( int i = 0; i < 5; i++ )
    ECE6745_CHECK_INT_EQ( a[i] , a_ref[i] );

  int b[]     = {5, 9, 1, 0, 13};
  int b_ref[] = {0, 1, 5, 9, 13};

  ubmark_sort_xcel( b, 5 );

  for ( int i = 0; i < 5; i++ )
    ECE6745_CHECK_INT_EQ( b[i] , b_ref[i] );

  int c[]     = {500, 16, 140, 32, 1000};
  int c_ref[] = {16, 32, 140, 500, 1000};

  ubmark_sort_xcel( c, 5 );

  for ( int i = 0; i < 5; i++ )
    ECE6745_CHECK_INT_EQ( c[i] , c_ref[i] );

  ECE6745_CHECK_INT_EQ( ece6745_get_heap_usage(), 0 );
}

//------------------------------------------------------------------------
// test_case_3_sort_empty
//------------------------------------------------------------------------
// A simple test case that tests sorting an empty array. Just make sure
// it doesn't segfault or crash or anything.

void test_case_3_sort_empty()
{
  ECE6745_CHECK( L"test_case_3_sort_empty" );

  int* a = NULL;

  ubmark_sort_xcel( a, 0 );

  int* b = NULL;

  ubmark_sort_xcel( b, 0 );

  // Need at least one check ... if we get here it means we didn't crash

  ECE6745_CHECK_TRUE( 1 );
  ECE6745_CHECK_INT_EQ( ece6745_get_heap_usage(), 0 );
}

//------------------------------------------------------------------------
// test_case_4_sort_one_element
//------------------------------------------------------------------------
// A simple test case that tests sorting a one-element array.

void test_case_4_sort_one_element()
{
  ECE6745_CHECK( L"test_case_4_sort_one_element" );

  int a[] = {3};

  ubmark_sort_xcel( a, 1 );

  ECE6745_CHECK_INT_EQ( a[0] , 3 );
  ECE6745_CHECK_INT_EQ( ece6745_get_heap_usage(), 0 );
}

//------------------------------------------------------------------------
// test_case_5_sort_two_element
//------------------------------------------------------------------------

void test_case_5_sort_two_element()
{
  ECE6745_CHECK( L"test_case_5_sort_two_element" );

  int a[] = { 4, 3 };

  ubmark_sort_xcel( a, 2 );

  ECE6745_CHECK_INT_EQ( a[0], 3 );
  ECE6745_CHECK_INT_EQ( a[1], 4 );

  int b[] = { 5, 9 };

  ubmark_sort_xcel( b, 2 );

  ECE6745_CHECK_INT_EQ( b[0], 5 );
  ECE6745_CHECK_INT_EQ( b[1], 9 );
  ECE6745_CHECK_INT_EQ( ece6745_get_heap_usage(), 0 );
}

//------------------------------------------------------------------------
// test_case_6_sort_almost_sorted
//------------------------------------------------------------------------

void test_case_6_sort_almost_sorted()
{
  ECE6745_CHECK( L"test_case_6_sort_almost_sorted" );

  int a[]     = { 4, 3, 5, 6, 10 };
  int a_ref[] = { 3, 4, 5, 6, 10 };

  ubmark_sort_xcel( a, 5 );

  for ( int i = 0; i < 5; i++ )
    ECE6745_CHECK_INT_EQ( a[i] , a_ref[i] );

  int b[]     = {0, 2, 9, 5, 13};
  int b_ref[] = {0, 2, 5, 9, 13};

  ubmark_sort_xcel( b, 5 );

  for ( int i = 0; i < 5; i++ )
    ECE6745_CHECK_INT_EQ( b[i] , b_ref[i] );

  ECE6745_CHECK_INT_EQ( ece6745_get_heap_usage(), 0 );
}

//------------------------------------------------------------------------
// test_case_7_sort_sorted
//------------------------------------------------------------------------

void test_case_7_sort_sorted()
{
  ECE6745_CHECK( L"test_case_7_sort_sorted" );

  int a[]     = { 3, 4, 5, 6, 10 };
  int a_ref[] = { 3, 4, 5, 6, 10 };

  ubmark_sort_xcel( a, 5 );

  for ( int i = 0; i < 5; i++ )
    ECE6745_CHECK_INT_EQ( a[i] , a_ref[i] );

  int b[]     = { 0, 2, 5, 9, 13 };
  int b_ref[] = { 0, 2, 5, 9, 13 };

  ubmark_sort_xcel( b, 5 );

  for ( int i = 0; i < 5; i++ )
    ECE6745_CHECK_INT_EQ( b[i] , b_ref[i] );

  int c[]     = { 2, 32, 140, 500, 1000 };
  int c_ref[] = { 2, 32, 140, 500, 1000 };

  ubmark_sort_xcel( c, 5 );

  for ( int i = 0; i < 5; i++ )
    ECE6745_CHECK_INT_EQ( c[i] , c_ref[i] );

  int d[]     = { 42, 42, 42, 42, 42 };
  int d_ref[] = { 42, 42, 42, 42, 42 };

  ubmark_sort_xcel( d, 5 );

  for ( int i = 0; i < 5; i++ )
    ECE6745_CHECK_INT_EQ( d[i] , d_ref[i] );

  ECE6745_CHECK_INT_EQ( ece6745_get_heap_usage(), 0 );
}

//------------------------------------------------------------------------
// test_case_8_sort_reversed
//------------------------------------------------------------------------

void test_case_8_sort_reversed()
{
  ECE6745_CHECK( L"test_case_8_sort_reversed" );

  int a[]     = {10, 6, 5, 4, 3};
  int a_ref[] = {3, 4, 5, 6, 10};

  ubmark_sort_xcel( a, 5 );

  for ( int i = 0; i < 5; i++ )
    ECE6745_CHECK_INT_EQ( a[i] , a_ref[i] );

  int b[]     = {13, 9, 5, 2, 1};
  int b_ref[] = {1, 2, 5, 9, 13};

  ubmark_sort_xcel( b, 5 );

  for ( int i = 0; i < 5; i++ )
    ECE6745_CHECK_INT_EQ( b[i] , b_ref[i] );

  int c[]     = {1000, 500, 140, 32, 1};
  int c_ref[] = {1, 32, 140, 500, 1000};

  ubmark_sort_xcel( c, 5 );

  for ( int i = 0; i < 5; i++ )
    ECE6745_CHECK_INT_EQ( c[i] , c_ref[i] );

  ECE6745_CHECK_INT_EQ( ece6745_get_heap_usage(), 0 );
}

//------------------------------------------------------------------------
// test_case_9_few_unique
//------------------------------------------------------------------------

void test_case_9_sort_few_unique()
{
  ECE6745_CHECK( L"test_case_9_sort_few_unique" );

  int a[]     = {3, 4, 4, 3, 4};
  int a_ref[] = {3, 3, 4, 4, 4};

  ubmark_sort_xcel( a, 5 );

  for ( int i = 0; i < 5; i++ )
    ECE6745_CHECK_INT_EQ( a[i] , a_ref[i] );

  int b[]     = {25, 5, 5, 5, 25};
  int b_ref[] = {5, 5, 5, 25, 25};

  ubmark_sort_xcel( b, 5 );

  for ( int i = 0; i < 5; i++ )
    ECE6745_CHECK_INT_EQ( b[i] , b_ref[i] );

  ECE6745_CHECK_INT_EQ( ece6745_get_heap_usage(), 0 );
}

//------------------------------------------------------------------------
// test_case_10_sort_random_many_4
//------------------------------------------------------------------------

void test_case_10_sort_random_many_4()
{
  ECE6745_CHECK( L"test_case_10_sort_random_many_4" );

  ece6745_srand(0x0000abcd);

  for ( int i = 0; i < 64; i++ ) {

    int a[4];

    for ( int i = 0; i < 4; i++ )
      a[i] = 0x0000ffff & ece6745_rand();

    ubmark_sort_xcel( a, 4 );

    ECE6745_CHECK_TRUE( is_sorted( a, 4 ) );
  }

  ECE6745_CHECK_INT_EQ( ece6745_get_heap_usage(), 0 );
}

//------------------------------------------------------------------------
// test_case_11_sort_random_64
//------------------------------------------------------------------------

void test_case_11_sort_random_64()
{
  ECE6745_CHECK( L"test_case_11_sort_random_64" );

  ece6745_srand(0x0000cafe);

  int  size  = 64;
  int* a = ece6745_malloc( size * (int) sizeof(int) );

  for ( int i = 0; i < size; i++ )
    a[i] = 0x0000ffff & ece6745_rand();

  ubmark_sort_xcel( a, size );

  ECE6745_CHECK_TRUE( is_sorted( a, 64 ) );

  ece6745_free( a );

  ECE6745_CHECK_INT_EQ( ece6745_get_heap_usage(), 0 );
}

//------------------------------------------------------------------------
// test_case_12_sort_random_67
//------------------------------------------------------------------------

void test_case_12_sort_random_67()
{
  ECE6745_CHECK( L"test_case_12_sort_random_67" );

  ece6745_srand(0x0000beef);

  int  size = 67;
  int* a = ece6745_malloc( size * (int) sizeof(int) );

  for ( int i = 0; i < size; i++ )
    a[i] = 0x0000ffff & ece6745_rand();

  ubmark_sort_xcel( a, size );

  ECE6745_CHECK_TRUE( is_sorted( a, 67 ) );

  ece6745_free( a );

  ECE6745_CHECK_INT_EQ( ece6745_get_heap_usage(), 0 );
}

//------------------------------------------------------------------------
// test_case_13_sort_eval_dataset
//------------------------------------------------------------------------

void test_case_13_sort_eval_dataset()
{
  ECE6745_CHECK( L"test_case_13_sort_eval_dataset" );

  ubmark_sort_xcel( eval_src, eval_size );

  ECE6745_CHECK_TRUE( is_sorted( eval_src, eval_size ) );
  ECE6745_CHECK_INT_EQ( ece6745_get_heap_usage(), 0 );
}

//------------------------------------------------------------------------
// main
//------------------------------------------------------------------------

int main( int argc, char** argv )
{
  __n = ( argc == 1 ) ? 0 : ece6745_atoi( argv[1] );

  if ( (__n <= 0) || (__n == 1 ) ) test_case_1_sort_basic();
  if ( (__n <= 0) || (__n == 2 ) ) test_case_2_sort_five_element();
  if ( (__n <= 0) || (__n == 3 ) ) test_case_3_sort_empty();
  if ( (__n <= 0) || (__n == 4 ) ) test_case_4_sort_one_element();
  if ( (__n <= 0) || (__n == 5 ) ) test_case_5_sort_two_element();
  if ( (__n <= 0) || (__n == 6 ) ) test_case_6_sort_almost_sorted();
  if ( (__n <= 0) || (__n == 7 ) ) test_case_7_sort_sorted();
  if ( (__n <= 0) || (__n == 8 ) ) test_case_8_sort_reversed();
  if ( (__n <= 0) || (__n == 9 ) ) test_case_9_sort_few_unique();
  if ( (__n <= 0) || (__n == 10) ) test_case_10_sort_random_many_4();
  if ( (__n <= 0) || (__n == 11) ) test_case_11_sort_random_64();
  if ( (__n <= 0) || (__n == 12) ) test_case_12_sort_random_67();
  if ( (__n <= 0) || (__n == 13) ) test_case_13_sort_eval_dataset();

  ece6745_wprintf( L"\n\n" );
  return ece6745_check_status;
}
