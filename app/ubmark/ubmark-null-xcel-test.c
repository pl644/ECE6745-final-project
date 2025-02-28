//========================================================================
// ubmark-null-xcel-test
//========================================================================
// This test assumes we have a NullXcel accelerator and illustrates how
// we can read and write the xr0 register in the NullXcel using inline
// assembly.

#include "ece6745.h"

//------------------------------------------------------------------------
// null_xcel
//------------------------------------------------------------------------

#ifdef _RISCV

__attribute__ ((noinline))
int ubmark_null_xcel( int in )
{
  int result;
  __asm__ (
    "csrw 0x7e0,     %[in];\n"
    "csrr %[result], 0x7e0;\n"

    // Outputs from the inline assembly block

    : [result] "=r"(result)

    // Inputs to the inline assembly block

    : [in] "r"(in)

  );
  return result;
}

#else

int ubmark_null_xcel( int in )
{
  return in;
}

#endif

//------------------------------------------------------------------------
// Test basic
//------------------------------------------------------------------------

void test_case_1_basic()
{
  ECE6745_CHECK( L"test_case_1_basic" );

  ECE6745_CHECK_INT_EQ( ubmark_null_xcel(  8 ),   8 );
  ECE6745_CHECK_INT_EQ( ubmark_null_xcel( -8 ),  -8 );
  ECE6745_CHECK_INT_EQ( ubmark_null_xcel( 42 ),  42 );
}

//------------------------------------------------------------------------
// main
//------------------------------------------------------------------------

int main( int argc, char** argv )
{
  __n = ( argc == 1 ) ? 0 : ece6745_atoi( argv[1] );

  if ( (__n <= 0) || (__n == 1) ) test_case_1_basic();

  ece6745_wprintf( L"\n\n" );
  return ece6745_check_status;
}

