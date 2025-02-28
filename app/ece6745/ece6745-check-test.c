//========================================================================
// Unit tests for unit testing framework
//========================================================================

#include "ece6745-misc.h"
#include "ece6745-check.h"
#include "ece6745-wprint.h"

//------------------------------------------------------------------------
// Test check true macro
//------------------------------------------------------------------------

void test_case_1_check_true()
{
  ECE6745_CHECK( L"test_case_1_check_true" );

  ECE6745_CHECK_TRUE( 1 == 1 );
  ECE6745_CHECK_TRUE( 1 != 2 );
}

//------------------------------------------------------------------------
// Test check false macro
//------------------------------------------------------------------------

void test_case_2_check_false()
{
  ECE6745_CHECK( L"test_case_2_check_false" );

  ECE6745_CHECK_FALSE( 1 != 1 );
  ECE6745_CHECK_FALSE( 1 == 2 );
}

//------------------------------------------------------------------------
// Test check int equal
//------------------------------------------------------------------------

void test_case_3_check_int_eq()
{
  ECE6745_CHECK( L"test_case_3_check_int_eq" );

  ECE6745_CHECK_INT_EQ( 1,   1 );
  ECE6745_CHECK_INT_EQ( 1+1, 2 );
}

//------------------------------------------------------------------------
// Test check note macro
//------------------------------------------------------------------------

void test_case_4_check_note()
{
  ECE6745_CHECK( L"test_case_4_check_note" );

  ECE6745_CHECK_NOTE( L"foo" );
  ECE6745_CHECK_NOTE( L"bar" );
  ECE6745_CHECK_NOTE( L"int in note (%d)",    1      );
  ECE6745_CHECK_NOTE( L"int in note (%d,%d)", 1, 2   );
  ECE6745_CHECK_NOTE( L"str in note (%S)",    L"foo" );
}

//------------------------------------------------------------------------
// main
//------------------------------------------------------------------------

int main( int argc, char** argv )
{
  __n = ( argc == 1 ) ? 0 : ece6745_atoi( argv[1] );

  if ( (__n <= 0) || (__n == 1) ) test_case_1_check_true();
  if ( (__n <= 0) || (__n == 2) ) test_case_2_check_false();
  if ( (__n <= 0) || (__n == 3) ) test_case_3_check_int_eq();
  if ( (__n <= 0) || (__n == 4) ) test_case_4_check_note();

  ece6745_wprintf( L"\n\n" );
  return ece6745_check_status;
}

