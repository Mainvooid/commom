#ifndef _COMMON_IPC_HPP_
#define _COMMON_IPC_HPP_

#include <cstddef>
#include <cstdint>
#include <limits>       // std::numeric_limits
#include <new>
#include <utility>
#include <tuple>
#include <vector>
#include <type_traits>
#include <Windows.h>

#include <common/windows.hpp>

/**
  @addtogroup common
  @{
    @defgroup ipc ipc - Inter-Process Communication
  @}
*/
namespace common {
    /// @addtogroup common
    /// @{
    namespace ipc {
        /// @addtogroup ipc
        /// @{
        /// @defgroup shm shm - Inter-Process Communication : shared memory
        namespace shm {
            /// @addtogroup ipc
            /// @{

            /// @}
        } // namespace shm
        /// @}
    } // namespace ipc
    /// @}
} // namespace common

#endif // _COMMON_IPC_HPP_