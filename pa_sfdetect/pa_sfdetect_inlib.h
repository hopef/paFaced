#pragma once


#ifdef _DEBUG
#pragma comment(lib, "pa_sfdetectd.lib")
#elif defined(NDEBUG)
#pragma comment(lib, "pa_sfdetect.lib")
#endif
