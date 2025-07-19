#ifndef LIFTED_DLLEXPORT_H
#define LIFTED_DLLEXPORT_H

#if !defined(LIFTED_SHARED_DEFINE)
#define LIFTED_DLLEXPORT
#define LIFTED_CONTRIB_DLLEXPORT
#define LIFTED_TEST_DLLEXPORT
#else  // !LIFTED_SHARED_DEFINE

#ifndef LIFTED_DLLEXPORT
#if defined(LIFTED_EXPORTS)
/* We are building this library */
#ifdef _WIN32
#define LIFTED_DLLEXPORT __declspec(dllexport)
#else
#define LIFTED_DLLEXPORT __attribute__((visibility("default")))
#endif
#else  // defined(LIFTED_EXPORTS)
/* We are using this library */
#ifdef _WIN32
#define LIFTED_DLLEXPORT __declspec(dllimport)
#else
#define LIFTED_DLLEXPORT __attribute__((visibility("default")))
#endif
#endif  // defined(LIFTED_EXPORTS)
#endif  // LIFTED_DLLEXPORT

#endif  // !LIFTED_SHARED_DEFINE

#endif /* LIFTED_DLLEXPORT_H */