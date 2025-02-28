//========================================================================
// Unit tests for ubmark mfilt
//========================================================================

#include "ece6745.h"
#include "ubmark-mfilt.h"
#include "ubmark-mfilt.dat"

//------------------------------------------------------------------------
// Test size 3x3 image
//------------------------------------------------------------------------

void test_case_1_image_3x3()
{
  ECE6745_CHECK( L"test_case_1_image_3x3" );

  int src[] =
  {
    11, 12, 13,
    21, 42, 23,
    31, 32, 33,
  };

  int mask[] =
  {
     1,  1,  1,
     1,  1,  1,
     1,  1,  1,
  };

  int dest[] =
  {
     1,  1,  1,
     1,  1,  1,
     1,  1,  1,
  };

  int ref[] =
  {
     0,  0,  0,
     0, 27,  0,
     0,  0,  0,
  };

  ubmark_mfilt( dest, mask, src, 3 );

  for ( int i = 0; i < 3*3; i++ )
    ECE6745_CHECK_INT_EQ( dest[i], ref[i] );
}

//------------------------------------------------------------------------
// Test size 4x4 image
//------------------------------------------------------------------------

void test_case_2_image_4x4()
{
  ECE6745_CHECK( L"test_case_2_image_4x4" );

  int src[] =
  {
    11, 12, 13, 14,
    21, 42, 24, 23,
    31, 53, 35, 33,
    41, 42, 43, 44,
  };

  int mask[] =
  {
     1,  1,  1,  1,
     1,  1,  1,  1,
     1,  1,  1,  1,
     1,  1,  1,  1,
  };

  int dest[] =
  {
     1,  1,  1,  1,
     1,  1,  1,  1,
     1,  1,  1,  1,
     1,  1,  1,  1,
  };

  int ref[] =
  {
     0,  0,  0,  0,
     0, 31, 27,  0,
     0, 41, 37,  0,
     0,  0,  0 , 0,
  };

  ubmark_mfilt( dest, mask, src, 4 );

  for ( int i = 0; i < 4*4; i++ )
    ECE6745_CHECK_INT_EQ( dest[i], ref[i] );
}

//------------------------------------------------------------------------
// Test size 4x4 image with mask
//------------------------------------------------------------------------

void test_case_3_image_4x4_masked()
{
  ECE6745_CHECK( L"test_case_2_image_4x4_masked" );

  int src[] =
  {
    11, 12, 13, 14,
    21, 42, 24, 23,
    31, 53, 35, 33,
    41, 42, 43, 44,
  };

  int mask[] =
  {
     0,  0,  0,  0,
     0,  1,  0,  0,
     0,  0,  1,  0,
     0,  0,  0,  0,
  };

  int dest[] =
  {
     1,  1,  1,  1,
     1,  1,  1,  1,
     1,  1,  1,  1,
     1,  1,  1,  1,
  };

  int ref[] =
  {
     0,  0,  0,  0,
     0, 31, 24,  0,
     0, 53, 37,  0,
     0,  0,  0 , 0,
  };

  ubmark_mfilt( dest, mask, src, 4 );

  for ( int i = 0; i < 4*4; i++ )
    ECE6745_CHECK_INT_EQ( dest[i], ref[i] );
}

//------------------------------------------------------------------------
// Test eval dataset
//------------------------------------------------------------------------

void test_case_4_eval_dataset()
{
  ECE6745_CHECK( L"test_case_4_eval_dataset" );

  int* dest = ece6745_malloc( eval_size*eval_size * (int)sizeof(int) );

  ubmark_mfilt( dest, eval_mask, eval_src, eval_size );

  for ( int i = 0; i < eval_size*eval_size; i++ )
    ECE6745_CHECK_INT_EQ( dest[i], eval_ref[i] );

  ece6745_free(dest);
}

//------------------------------------------------------------------------
// main
//------------------------------------------------------------------------

int main( int argc, char** argv )
{
  __n = ( argc == 1 ) ? 0 : ece6745_atoi( argv[1] );

  if ( (__n <= 0) || (__n == 1) ) test_case_1_image_3x3();
  if ( (__n <= 0) || (__n == 2) ) test_case_2_image_4x4();
  if ( (__n <= 0) || (__n == 3) ) test_case_3_image_4x4_masked();
  if ( (__n <= 0) || (__n == 4) ) test_case_4_eval_dataset();

  ece6745_wprintf( L"\n\n" );
  return ece6745_check_status;
}

