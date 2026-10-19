#ifndef COGNI_COMMON_EXPORT_HPP
#define COGNI_COMMON_EXPORT_HPP

#if defined(_WIN32) || defined(__CYGWIN__)
#  define COGNI_EXPORT __declspec(dllexport)
#  define COGNI_IMPORT __declspec(dllimport)
#elif defined(__GNUC__) || defined(__clang__)
#  define COGNI_EXPORT __attribute__((visibility("default")))
#  define COGNI_IMPORT __attribute__((visibility("default")))
#else
#  define COGNI_EXPORT
#  define COGNI_IMPORT
#endif

#ifndef COGNICAR_INTERFACE_API
#  if defined(COGNICAR_INTERFACE_STATIC)
#    define COGNICAR_INTERFACE_API
#  elif defined(COGNICAR_INTERFACE_BUILDING)
#    define COGNICAR_INTERFACE_API COGNI_EXPORT
#  else
#    define COGNICAR_INTERFACE_API COGNI_IMPORT
#  endif
#endif

#ifndef TRAJECTORY_TRACKING_API
#  if defined(TRAJECTORY_TRACKING_STATIC)
#    define TRAJECTORY_TRACKING_API
#  elif defined(TRAJECTORY_TRACKING_BUILDING)
#    define TRAJECTORY_TRACKING_API COGNI_EXPORT
#  else
#    define TRAJECTORY_TRACKING_API COGNI_IMPORT
#  endif
#endif

#ifndef LOW_LEVEL_CONTROL_API
#  if defined(LOW_LEVEL_CONTROL_STATIC)
#    define LOW_LEVEL_CONTROL_API
#  elif defined(LOW_LEVEL_CONTROL_BUILDING)
#    define LOW_LEVEL_CONTROL_API COGNI_EXPORT
#  else
#    define LOW_LEVEL_CONTROL_API COGNI_IMPORT
#  endif
#endif

#ifndef STATE_ESTIMATOR_API
#  if defined(STATE_ESTIMATOR_STATIC)
#    define STATE_ESTIMATOR_API
#  elif defined(STATE_ESTIMATOR_BUILDING)
#    define STATE_ESTIMATOR_API COGNI_EXPORT
#  else
#    define STATE_ESTIMATOR_API COGNI_IMPORT
#  endif
#endif

#endif
