//========================================================================
// Unit tests for ubmark vvadd
//========================================================================

#include "ece6745.h"
#include "ubmark-vvadd-xcel.h"
#include "ubmark-vvadd.dat"

//------------------------------------------------------------------------
// Test positive inputs
//------------------------------------------------------------------------

void test_case_1_pos()
{
  ECE6745_CHECK( L"test_case_1_pos" );

  int src0[] = {  1,  2,  3,  4 };
  int src1[] = {  5,  6,  7,  8 };
  int dest[] = {  0,  0,  0,  0 };
  int ref[]  = {  6,  8, 10, 12 };

  ubmark_vvadd_xcel( dest, src0, src1, 4 );

  for ( int i = 0; i < 4; i++ )
    ECE6745_CHECK_INT_EQ( dest[i], ref[i] );
}

//------------------------------------------------------------------------
// Test negative inputs
//------------------------------------------------------------------------

void test_case_2_neg()
{
  ECE6745_CHECK( L"test_case_2_neg" );

  int src0[] = { -1, -2, -3,  4 };
  int src1[] = {  5, -6,  7,  8 };
  int dest[] = {  0,  0,  0,  0 };
  int ref[]  = {  4, -8,  4, 12 };

  ubmark_vvadd_xcel( dest, src0, src1, 4 );

  for ( int i = 0; i < 4; i++ )
    ECE6745_CHECK_INT_EQ( dest[i], ref[i] );
}

//------------------------------------------------------------------------
// Test size 9
//------------------------------------------------------------------------

void test_case_3_size9()
{
  ECE6745_CHECK( L"test_case_3_size9" );

  int src0[9];
  int src1[9];
  int dest[9];
  int ref[9];

  for ( int i = 0; i < 9; i++ ) {
    src0[i] = i;
    src1[i] = 2*i;
    dest[i] = 0;
    ref[i]  = i + 2*i;
  }

  ubmark_vvadd_xcel( dest, src0, src1, 9 );

  for ( int i = 0; i < 9; i++ )
    ECE6745_CHECK_INT_EQ( dest[i], ref[i] );
}

//------------------------------------------------------------------------
// Test size 16
//------------------------------------------------------------------------

void test_case_4_size16()
{
  ECE6745_CHECK( L"test_case_4_size16" );

  int src0[16];
  int src1[16];
  int dest[16];
  int ref[16];

  for ( int i = 0; i < 16; i++ ) {
    src0[i] = i;
    src1[i] = 2*i;
    dest[i] = 0;
    ref[i]  = i + 2*i;
  }

  ubmark_vvadd_xcel( dest, src0, src1, 16 );

  for ( int i = 0; i < 16; i++ )
    ECE6745_CHECK_INT_EQ( dest[i], ref[i] );
}

//------------------------------------------------------------------------
// Test eval dataset
//------------------------------------------------------------------------

void test_case_5_eval_dataset()
{
  ECE6745_CHECK( L"test_case_5_eval_dataset" );

  int* dest = ece6745_malloc( eval_size * (int)sizeof(int) );

  ubmark_vvadd_xcel( dest, eval_src0, eval_src1, eval_size );

  for ( int i = 0; i < eval_size; i++ )
    ECE6745_CHECK_INT_EQ( dest[i], eval_ref[i] );

  ece6745_free(dest);
}

//------------------------------------------------------------------------
// main
//------------------------------------------------------------------------

int main( int argc, char** argv )
{
  __n = ( argc == 1 ) ? 0 : ece6745_atoi( argv[1] );

  if ( (__n <= 0) || (__n == 1) ) test_case_1_pos();
  if ( (__n <= 0) || (__n == 2) ) test_case_2_neg();
  if ( (__n <= 0) || (__n == 3) ) test_case_3_size9();
  if ( (__n <= 0) || (__n == 4) ) test_case_4_size16();
  if ( (__n <= 0) || (__n == 5) ) test_case_5_eval_dataset();

  ece6745_wprintf( L"\n\n" );
  return ece6745_check_status;
}

