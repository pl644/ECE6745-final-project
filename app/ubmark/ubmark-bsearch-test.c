//========================================================================
// Unit tests for ubmark bsearch
//========================================================================

#include "ece6745.h"
#include "ubmark-bsearch.h"
#include "ubmark-bsearch.dat"

//------------------------------------------------------------------------
// Test basic
//------------------------------------------------------------------------

void test_case_1_basic()
{
  ECE6745_CHECK( L"test_case_1_basic" );

  int sorted_keys[] = { 1, 3, 5, 7, 9 };
  int search_keys[] = { 3 };
  int idxs[]        = { 0 };

  ubmark_bsearch( idxs, search_keys, 1, sorted_keys, 5 );

  ECE6745_CHECK_INT_EQ( idxs[0], 1 );
}

//------------------------------------------------------------------------
// Test basic no match
//------------------------------------------------------------------------

void test_case_2_basic_nomatch()
{
  ECE6745_CHECK( L"test_case_2_basic_nomatch" );

  int sorted_keys[] = { 1, 3, 5, 7, 9 };
  int search_keys[] = { 4 };
  int idxs[]        = { 0 };

  ubmark_bsearch( idxs, search_keys, 1, sorted_keys, 5 );

  ECE6745_CHECK_INT_EQ( idxs[0], -1 );
}

//------------------------------------------------------------------------
// Test size 5
//------------------------------------------------------------------------

void test_case_3_size5()
{
  ECE6745_CHECK( L"test_case_3_size5" );

  int sorted_keys[] = { 1, 3, 5, 7, 9 };

  int search_keys[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10 };
  int idxs[]        = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  int ref[]         = {-1, 0,-1, 1,-1, 2,-1, 3,-1, 4,-1 };

  ubmark_bsearch( idxs, search_keys, 11, sorted_keys, 5 );

  for ( int i = 0; i < 11; i++ )
    ECE6745_CHECK_INT_EQ( idxs[i], ref[i] );
}

//------------------------------------------------------------------------
// Test sizs 16
//------------------------------------------------------------------------

void test_case_4_size16()
{
  ECE6745_CHECK( L"test_case_4_size16" );

  int sorted_keys[16];
  int search_keys[33];
  int idxs[33];
  int ref[33];

  int j = 0;
  for ( int i = 0; i < 33; i++ ) {
    search_keys[i] = i;
    idxs[i]        = 0;

    // if i is odd, add this key to sorted keys
    if ( i % 2 == 1 ) {
      ref[i] = j;
      sorted_keys[j] = i;
      j += 1;
    }

    // if i is even, then do not add to sorted keys
    else {
      ref[i] = -1;
    }
  }

  ubmark_bsearch( idxs, search_keys, 33, sorted_keys, 16 );

  for ( int i = 0; i < 33; i++ )
    ECE6745_CHECK_INT_EQ( idxs[i], ref[i] );
}

//------------------------------------------------------------------------
// Test sizs 17
//------------------------------------------------------------------------
// Non-power of two is good to test

void test_case_5_size17()
{
  ECE6745_CHECK( L"test_case_5_size17" );

  int sorted_keys[17];
  int search_keys[35];
  int idxs[35];
  int ref[35];

  int j = 0;
  for ( int i = 0; i < 35; i++ ) {
    search_keys[i] = i;
    idxs[i]        = 0;

    // if i is odd, add this key to sorted keys
    if ( i % 2 == 1 ) {
      ref[i] = j;
      sorted_keys[j] = i;
      j += 1;
    }

    // if i is even, then do not add to sorted keys
    else {
      ref[i] = -1;
    }
  }

  ubmark_bsearch( idxs, search_keys, 35, sorted_keys, 17 );

  for ( int i = 0; i < 35; i++ )
    ECE6745_CHECK_INT_EQ( idxs[i], ref[i] );
}

//------------------------------------------------------------------------
// Test eval dataset
//------------------------------------------------------------------------

void test_case_6_eval_dataset()
{
  ECE6745_CHECK( L"test_case_6_eval_dataset" );

  int* idxs = ece6745_malloc( eval_search_keys_size * (int)sizeof(int) );

  ubmark_bsearch( idxs, eval_search_keys, eval_search_keys_size,
                  eval_sorted_keys, eval_sorted_keys_size );

  for ( int i = 0; i < eval_search_keys_size; i++ )
    ECE6745_CHECK_INT_EQ( idxs[i], eval_ref[i] );

  ece6745_free(idxs);
}

//------------------------------------------------------------------------
// main
//------------------------------------------------------------------------

int main( int argc, char** argv )
{
  __n = ( argc == 1 ) ? 0 : ece6745_atoi( argv[1] );

  if ( (__n <= 0) || (__n == 1) ) test_case_1_basic();
  if ( (__n <= 0) || (__n == 2) ) test_case_2_basic_nomatch();
  if ( (__n <= 0) || (__n == 3) ) test_case_3_size5();
  if ( (__n <= 0) || (__n == 4) ) test_case_4_size16();
  if ( (__n <= 0) || (__n == 5) ) test_case_5_size17();
  if ( (__n <= 0) || (__n == 6) ) test_case_6_eval_dataset();

  ece6745_wprintf( L"\n\n" );
  return ece6745_check_status;
}

