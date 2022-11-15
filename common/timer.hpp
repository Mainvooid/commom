#ifndef __TIMER_HPP__
#define __TIMER_HPP__

#include <thread>
#include <mutex>
#include <chrono>
#include <functional>
#include <vector>
#include <condition_variable>
#include <atomic>
#include <algorithm>

/**
  @addtogroup common
  @{
    @defgroup timer timer - simple timer
  @}
*/
namespace common
{
    /// @addtogroup common
    /// @{

    struct user_timer_obj : std::enable_shared_from_this<user_timer_obj>
    {
        std::size_t id;                 /**< 定时器ID */
        std::chrono::milliseconds dur;  /**< 定时器时间间隔 */
        std::function<void(void)> func; /**< 定时执行函数,TODO:可扩展任意函数 */

        user_timer_obj(std::size_t _id,
                       std::chrono::milliseconds _dur,
                       std::function<void(void)> _func)
            : id(_id), dur(_dur), func(_func) {}
    };

    /**
     * @brief 简单的跨平台计时器
     * @see test/test_timer.cpp
    */
    class timer
    {
    public:
        /**
        * @param 设置定时器最大刷新间隔 ms
        */
        timer(size_t next_flush_max = 50)
        {
            _is_init = false;
            set_next_flush_max(next_flush_max);
        };
        ~timer()
        {
            _release();
        };

        /**
         * @brief 启动一个定时器
         * @param _id 定时器ID
         * @param _dur 定时器时间间隔
         * @param _func 定时执行函数
        */
        bool start_timer(std::size_t _id,
                         std::chrono::milliseconds _dur,
                         std::function<void(void)> _func)
        {
            std::shared_ptr<user_timer_obj> timer_obj_ptr = std::make_shared<user_timer_obj>(_id, _dur, _func);
            return start_timer(std::move(timer_obj_ptr));
        };
        /**< @override */
        bool start_timer(std::shared_ptr<user_timer_obj> timer_obj_ptr)
        {
            _init();
            for (auto &task : _timer_task)
            {
                if (task.first->id == timer_obj_ptr->id)
                {
                    return false;
                }
            }
            std::unique_lock<std::mutex> _timer_task_ul(_timer_task_mu);
            _timer_task.push_back({timer_obj_ptr, std::chrono::steady_clock::now()});
            return true;
        };

        void stop_timer(std::size_t timer_id)
        {
            if (_timer_task.empty())
                return;
            for (auto it = _timer_task.begin(); it != _timer_task.end();)
            {
                if (it->first->id == timer_id)
                {
                    std::unique_lock<std::mutex> _timer_task_ul(_timer_task_mu);
                    it = _timer_task.erase(it);
                    break;
                }
                else
                {
                    it++;
                }
            }
            if (_timer_task.empty())
                _release(); //若已删除最后一个定时任务，释放线程
        };
        void stop_timer(std::shared_ptr<user_timer_obj> timer_obj_ptr)
        {
            stop_timer(timer_obj_ptr->id);
        };

        void set_next_flush_max(size_t next_flush_max)
        {
            _next_flush_max = next_flush_max;
        };

    private:
        void _init()
        {
            if (_is_init)
                return;
            _next_flush_duration = std::chrono::milliseconds(_next_flush_max);
            _is_run.store(true);
            _worker_thread = std::thread(std::bind(&timer::_worker, this));
            _worker_thread.detach();
            _observer_thread = std::thread(std::bind(&timer::_observer, this));
            _observer_thread.detach();
            _is_init = true;
        };
        void _release()
        {
            _is_run.store(false);
            _cond.notify_all();
            _observer_thread.~thread();
            _worker_thread.~thread();
            _is_init = false;
            std::unique_lock<std::mutex> _timer_task_ul(_timer_task_mu);
            _timer_task.clear();
        };

        /**
         * @brief 获取需要执行的定时任务id
        */
        void _flush_tasks()
        {
            using namespace std::chrono;
            if (_timer_task.empty())
                return;

            steady_clock::time_point now = steady_clock::now();
            for (auto &task : _timer_task)
            {
                steady_clock::time_point &old = task.second;
                milliseconds &dur = task.first->dur;
                //检测是否触发
                milliseconds time_span = duration_cast<milliseconds>(now - old);
                if (time_span.count() >= dur.count())
                {
                    old = now;
                    std::size_t &id = task.first->id;

                    auto iter = std::find(_activated_task_ids.begin(), _activated_task_ids.end(), id);
                    if (iter == _activated_task_ids.end())
                    { //not found
                        std::unique_lock<std::mutex> _activated_task_ul(_activated_task_mu);
                        _activated_task_ids.push_back(id);
                    }
                }
            }
        };

        /**
         * @brief 获取下一次刷新间隔
        */
        void _get_next_flush_duration()
        {
            using namespace std::chrono;

            if (_timer_task.empty())
            {
                return;
            }
            //重置刷新时间
            _next_flush_duration = milliseconds(_next_flush_max);
            steady_clock::time_point now = steady_clock::now();
            for (auto &task : _timer_task)
            {
                steady_clock::time_point &old = task.second;
                milliseconds &dur = task.first->dur;
                milliseconds time_span = duration_cast<duration<size_t>>(now - old);
                //检测间隔不大于_NEXT_FLUSH_MAX
                milliseconds next;
                if (time_span.count() >= dur.count())
                {
                    //工作线程可能还未开始处理,下次刷新时间为定时时间dur
                    next = dur;
                }
                else
                {
                    next = dur - time_span;
                }
                _next_flush_duration = std::min(_next_flush_duration, next);
            }
        };

        /**
         * @brief 工作线程方法
        */
        void _worker()
        {
            while (_is_run.load())
            {
                std::unique_lock<std::mutex> ul(_mu);
                while (_activated_task_ids.empty()) // 无定时任务
                {
                    _cond.wait(ul); // 非忙等
                }
                //找到id绑定的函数并执行
                size_t size = _activated_task_ids.size();
                while (size--)
                {
                    size_t &id = _activated_task_ids.front();
                    for (auto &task : _timer_task)
                    {
                        if (task.first->id == id)
                        {
                            task.first->func();
                            std::unique_lock<std::mutex> _activated_task_ul(_activated_task_mu);
                            _activated_task_ids.erase(_activated_task_ids.begin());
                            break;
                        }
                    }
                }
            }
        };

        /**
         * @brief 观察线程方法
        */
        void _observer()
        {
            while (_is_run.load())
            {
                std::unique_lock<std::mutex> _ul(_mu);
                _flush_tasks();
                if (!_activated_task_ids.empty())
                {
                    _cond.notify_all();
                    std::this_thread::yield(); //尽管交出时间片，系统不一定立即调用wocker线程
                }
                else
                {
                    //没有事件机制只能轮询自检,为避免忙等,需要计算下次刷新时间,同时为了及时响应新的计时任务,有个最大等待时间
                    _get_next_flush_duration();
                    std::this_thread::sleep_for(_next_flush_duration);
                }
            }
        };

    private:
        bool _is_init;                 /**< 线程是否初始化 */
        std::thread _worker_thread;    /**< 工作线程,TODO:可扩展线程池 */
        std::thread _observer_thread;  /**< 观察者线程 */
        std::mutex _mu;                /**< 线程锁 */
        std::condition_variable _cond; /**< 条件变量 */
        std::atomic<bool> _is_run;     /**< 控制线程退出 */
        std::vector<std::pair<std::shared_ptr<user_timer_obj>,
                              std::chrono::time_point<std::chrono::steady_clock>>>
            _timer_task;                              /**< 定时任务 */
        std::vector<std::size_t> _activated_task_ids; /**< 激活的定时任务ID组 */
        std::mutex _activated_task_mu;                /**< 访问锁 */
        std::mutex _timer_task_mu;                    /**< 访问锁 */

        size_t _next_flush_max;                         /**< 刷新间隔不大于此数 ms*/
        std::chrono::milliseconds _next_flush_duration; /**< 下一次刷新间隔 */
    };
    /// @}
} // namespace common
#endif // __TIMER_HPP__