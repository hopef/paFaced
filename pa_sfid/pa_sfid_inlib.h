#pragma once


#ifdef _DEBUG
#pragma comment(lib, "pa_sfidd.lib")
#elif defined(NDEBUG)
#pragma comment(lib, "pa_sfid.lib")
#endif
