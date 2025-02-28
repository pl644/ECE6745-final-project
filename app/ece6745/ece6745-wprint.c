//========================================================================
// ece6745-wprint
//========================================================================

#include "ece6745-wprint.h"

//------------------------------------------------------------------------
// ece6745_wprintf
//------------------------------------------------------------------------

#ifdef _RISCV

#include "ece6745-wprint.h"
#include <stdarg.h>

void ece6745_wprintf( const wchar_t* fmt, ... )
{
  va_list args;
  va_start(args, fmt);

  int flag = 0;
  while (*fmt != '\0') {
    if (*fmt == '%' ) {
      flag = 1;
    }
    else if ( flag && (*fmt == 'd') ) {
      ece6745_wprint_int( va_arg(args, int) );
      flag = 0;
    }
    else if ( flag && (*fmt == 'C') ) {
      // note automatic conversion to integral type
      ece6745_wprint_char( (wchar_t) (va_arg(args, int)) );
      flag = 0;
    }
    else if ( flag && (*fmt == 'S') ) {
      ece6745_wprint_str( va_arg(args, wchar_t*) );
      flag = 0;
    }
    else {
      ece6745_wprint_char( *fmt );
    }
    ++fmt;
  }
  va_end(args);
}

#else

// We always need at least something in an object file, otherwise
// the native build won't work. So we create a dummy function for native
// builds.

int ece6745_( int arg )
{
  return arg;
}

#endif

