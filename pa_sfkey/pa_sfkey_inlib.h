#pragma once

#ifdef _DEBUG
#pragma comment(lib, "pa_sfkeyd.lib")
#elif defined(NDEBUG)
#pragma comment(lib, "pa_sfkey.lib")
#endif
