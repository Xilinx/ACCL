/*******************************************************************************
#  Copyright (C) 2022 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#
*******************************************************************************/

#pragma once

#include <ctime>
#include <map>
#include <mutex>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

enum class log_level {
    error = 0,
    critical_warning = 1,
    warning = 2,
    info = 3,
    verbose = 4,
    debug = 5
};

const std::map<log_level, std::string> LOG_STRING = {
    {log_level::error, "ERROR"},
    {log_level::critical_warning, "CRITICAL WARNING"},
    {log_level::warning, "WARNING"},
    {log_level::info, "INFO"},
    {log_level::verbose, "VERBOSE"},
    {log_level::debug, "DEBUG"}
};

class Log {
public:
    Log() {}

    Log(log_level level, unsigned int rank, std::ostream &ostream = std::cout)
            : _level(level), _ostream(&ostream), _local_rank(rank) {
        buffer.str("");
        buffer_level = log_level::info;
    }

    Log(const Log &log) : Log(log.level(),log.rank(), *log.ostream()) {}

    log_level level() const {
        return _level;
    }

    unsigned int rank() const {
        return _local_rank;
    }

    std::ostream *ostream() const {
        return _ostream;
    }

    void operator()(const std::string &message, log_level level) {
        if (level <= _level) {
            auto time = std::time(nullptr);
            auto local_time = *std::localtime(&time);
            char time_buffer[9];
            strftime(time_buffer, sizeof(time_buffer), "%H:%M:%S", &local_time);
            std::lock_guard<std::mutex> guard(logging_mutex);
            *_ostream << std::right << "[Rank " << std::setw(3) << std::to_string(_local_rank) << ": " 
                  << std::left  << std::setw(8) << LOG_STRING.at(level);
            *_ostream << " " << time_buffer << "] " << message;
            _ostream->flush();
        }
    }

    // Set log level of stream
    Log &operator<<(log_level level) {
        buffer_level = level;
        return *this;
    }

    // Intercept std::endl
    Log &operator<<(std::ostream &(*os)(std::ostream &)) {
        if (buffer_level > _level) {
            return *this;
        }

        buffer << os;

        if (os == (std::basic_ostream<char>& (*)(std::basic_ostream<char>&)) &std::endl) {
            (*this)(buffer.str(), buffer_level);
            buffer.str("");
            buffer_level = log_level::info;
        }

        return *this;
    }

    template <typename T> Log &operator<<(const T &message) {
        if (buffer_level <= _level) {
            buffer << message;
        }
        return *this;
    }

    Log& operator=(const Log &l) {
        _level = l.level();
        _ostream = l.ostream();
        _local_rank = l.rank();
        buffer.str("");
        buffer_level = log_level::info;
        return *this;
    }

private:
    log_level _level;
    unsigned int _local_rank;
    std::ostream *_ostream;
    log_level buffer_level;
    std::ostringstream buffer;
    std::mutex logging_mutex;
};
