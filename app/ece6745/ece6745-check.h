//========================================================================
// ece6745-check
//========================================================================
// Very simple unit testing framework.
//
// _RISCV will only be defined when cross-compiling for RISCV.

#ifndef ECE6745_CHECK_H
#define ECE6745_CHECK_H

#include "ece6745-wprint.h"

//------------------------------------------------------------------------
// Helper Macros
//------------------------------------------------------------------------

// Colorize text

#define ECE6745_RED(x)    L"\033[31m" x L"\033[0m"
#define ECE6745_GREEN(x)  L"\033[32m" x L"\033[0m"
#define ECE6745_YELLOW(x) L"\033[33m" x L"\033[0m"

// Convert strings from ASCII to wide strings

#define ECE6745_A2W_(x)          L ## x
#define ECE6745_A2W(x)           ECE6745_A2W_(x)
#define ECE6745_STRINGIFY_(x)    #x
#define ECE6745_STRINGIFY(x)     ECE6745_STRINGIFY_(x)
#define ECE6745_STRINGIFY_A2W(x) ECE6745_A2W(ECE6745_STRINGIFY(x))

#define ECE6745_WFILE ECE6745_A2W(__FILE__)
#define ECE6745_WLINE ECE6745_STRINGIFY_A2W(__LINE__))

//------------------------------------------------------------------------
// Global Variables
//------------------------------------------------------------------------

// Which test case to run

extern int __n;

// Status of test file. Failed check sets this variable to 1.

extern int ece6745_check_status;

// Temporary variables to save the condition so that we don't evaluate
// the given expressions multiple times.

extern int ece6745_check_expr0;
extern int ece6745_check_expr1;

//------------------------------------------------------------------------
// ECE6745_CHECK
//------------------------------------------------------------------------
// Test case header

#define ECE6745_CHECK( ... )                                            \
  do {                                                                  \
    ece6745_wprintf( L"\n" ECE6745_WFILE L"::" __VA_ARGS__ L" " );      \
    if ( __n < 0 ) {                                                    \
      return;                                                           \
    }                                                                   \
  } while ( 0 )

//------------------------------------------------------------------------
// ECE6745_CHECK_FAIL
//------------------------------------------------------------------------
// Unconditionally fail a test case.

#define ECE6745_CHECK_FAIL( ... )                                       \
  do {                                                                  \
                                                                        \
    ece6745_wprint_str( ECE6745_RED(L"FAILED") );                       \
    if ( __n > 0 ) {                                                    \
      ece6745_wprintf( L"\n - [ " ECE6745_RED(L"FAILED") L" ] %S:%d: %S", \
        ECE6745_WFILE, __LINE__ );                                      \
      ece6745_wprintf( __VA_ARGS__ );                                   \
    }                                                                   \
    ece6745_flush();                                                    \
    ece6745_check_status = 1;                                           \
    return;                                                             \
                                                                        \
  } while ( 0 )

//------------------------------------------------------------------------
// ECE6745_CHECK_TRUE
//------------------------------------------------------------------------

#define ECE6745_CHECK_TRUE( expr_ )                                     \
  do {                                                                  \
    ece6745_check_expr0 = (expr_);                                      \
                                                                        \
    if ( !ece6745_check_expr0 ) {                                       \
      if ( __n > 0 ) {                                                  \
        ece6745_wprintf( L"\n - [ " ECE6745_RED(L"FAILED") L" ] %S:%d: %S (%d)", \
          ECE6745_WFILE, __LINE__,                                      \
          ECE6745_STRINGIFY_A2W(expr_), ece6745_check_expr0 );          \
      }                                                                 \
      else                                                              \
        ece6745_wprint_str( ECE6745_RED(L"FAILED") );                   \
      ece6745_flush();                                                  \
      ece6745_check_status = 1;                                         \
      return;                                                           \
    }                                                                   \
    else if ( __n > 0 ) {                                               \
      ece6745_wprintf( L"\n - [ " ECE6745_GREEN(L"passed") L" ] %S:%d: %S (%d)",\
        ECE6745_WFILE, __LINE__,                                        \
        ECE6745_STRINGIFY_A2W(expr_), ece6745_check_expr0 );            \
      ece6745_flush();                                                  \
    }                                                                   \
    else {                                                              \
      ece6745_wprint_str( ECE6745_GREEN(L".") );                        \
      ece6745_flush();                                                  \
    }                                                                   \
                                                                        \
  } while ( 0 )

//------------------------------------------------------------------------
// ECE6745_CHECK_FALSE
//------------------------------------------------------------------------

#define ECE6745_CHECK_FALSE( expr_ )                                    \
  do {                                                                  \
    ece6745_check_expr0 = (expr_);                                      \
                                                                        \
    if ( ece6745_check_expr0 ) {                                        \
      if ( __n > 0 ) {                                                  \
        ece6745_wprintf( L"\n - [ " ECE6745_RED(L"FAILED") L" ] %S:%d: %S (%d)", \
          ECE6745_WFILE, __LINE__,                                      \
          ECE6745_STRINGIFY_A2W(expr_), ece6745_check_expr0 );          \
      }                                                                 \
      else                                                              \
        ece6745_wprint_str( ECE6745_RED(L"FAILED") );                   \
      ece6745_flush();                                                  \
      ece6745_check_status = 1;                                         \
      return;                                                           \
    }                                                                   \
    else if ( __n > 0 ) {                                               \
      ece6745_wprintf( L"\n - [ " ECE6745_GREEN(L"passed") L" ] %S:%d: %S (%d)",\
        ECE6745_WFILE, __LINE__,                                        \
        ECE6745_STRINGIFY_A2W(expr_), ece6745_check_expr0 );            \
      ece6745_flush();                                                  \
    }                                                                   \
    else {                                                              \
      ece6745_wprint_str( ECE6745_GREEN(L".") );                        \
      ece6745_flush();                                                  \
    }                                                                   \
                                                                        \
  } while ( 0 )

//------------------------------------------------------------------------
// ECE6745_CHECK_INT_EQ
//------------------------------------------------------------------------
// Check to see if two expressions are equal

#define ECE6745_CHECK_INT_EQ( expr0_, expr1_ )                          \
  do {                                                                  \
    ece6745_check_expr0 = (expr0_);                                     \
    ece6745_check_expr1 = (expr1_);                                     \
                                                                        \
    if ( ece6745_check_expr0 != ece6745_check_expr1 ) {                 \
      if ( __n > 0 ) {                                                  \
        ece6745_wprintf( L"\n - [ " ECE6745_RED(L"FAILED") L" ] %S:%d: %S != %S (%d != %d)", \
          ECE6745_WFILE, __LINE__,                                      \
          ECE6745_STRINGIFY_A2W(expr0_), ECE6745_STRINGIFY_A2W(expr1_), \
          ece6745_check_expr0, ece6745_check_expr1 );                   \
      }                                                                 \
      else                                                              \
        ece6745_wprint_str( ECE6745_RED(L"FAILED") );                   \
      ece6745_flush();                                                  \
      ece6745_check_status = 1;                                         \
      return;                                                           \
    }                                                                   \
    else if ( __n > 0 ) {                                               \
      ece6745_wprintf( L"\n - [ " ECE6745_GREEN(L"passed") L" ] %S:%d: %S == %S (%d == %d)", \
        ECE6745_WFILE, __LINE__,                                        \
        ECE6745_STRINGIFY_A2W(expr0_), ECE6745_STRINGIFY_A2W(expr1_),   \
        ece6745_check_expr0, ece6745_check_expr1 );                     \
      ece6745_flush();                                                  \
    }                                                                   \
    else {                                                              \
      ece6745_wprint_str( ECE6745_GREEN(L".") );                        \
      ece6745_flush();                                                  \
    }                                                                   \
                                                                        \
  } while ( 0 )

//------------------------------------------------------------------------
// ECE6745_CHECK_NOTE
//------------------------------------------------------------------------

#define ECE6745_CHECK_NOTE( ... )                                       \
  do {                                                                  \
                                                                        \
    if ( __n > 0 ) {                                                    \
      ece6745_wprintf( L"\n - [ " ECE6745_YELLOW(L"-note-") L" ] %S:%d: ", \
        ECE6745_WFILE, __LINE__ );                                      \
      ece6745_wprintf( __VA_ARGS__ );                                   \
      ece6745_flush();                                                  \
    }                                                                   \
                                                                        \
  } while ( 0 )

#endif /* ECE6745_CHECK_H */

