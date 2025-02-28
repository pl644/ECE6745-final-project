//========================================================================
// example of using wprint
//========================================================================

#include "ece6745-wprint.h"

const wchar_t* str = L"foobar";
int b = 0;
int a[] = { 1,2,3 };

int main( void )
{
  // Print some numbers and characters

  ece6745_wprint_int ( 42    );
  ece6745_wprint_char( L' '  );
  ece6745_wprint_int ( 13    );
  ece6745_wprint_char( L' '  );
  ece6745_wprint_int ( 57    );
  ece6745_wprint_char( L'\n' );

  // Print some strings

  ece6745_wprint_str( L"Testing print function: \n" );
  ece6745_wprint_str( L"Hello, World!\n" );

  // Use printf

  ece6745_wprintf( L"number = %d\n", 42 );
  ece6745_wprintf( L"char   = %C\n", L'a' );
  ece6745_wprintf( L"string = %S\n", L"foobar" );

  ece6745_wprintf( L"number = %d, char = %C, string = %S \n", 42, L'a', L"foobar" );

  return 0;
}

