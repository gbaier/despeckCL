/* Copyright 2015, 2016 Gerald Baier
 *
 * This file is part of despeckCL.
 *
 * despeckCL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * despeckCL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with despeckCL. If not, see <http://www.gnu.org/licenses/>.
 */

#include "logging.h"

#include "easylogging++.h"
INITIALIZE_EASYLOGGINGPP

#include <map>

const std::map<const std::string, const el::Level> loglevel_map {
                                                                  {"debug",   el::Level::Debug},
                                                                  {"info",    el::Level::Info},
                                                                  {"verbose", el::Level::Verbose},
                                                                  {"warning", el::Level::Warning},
                                                                  {"fatal",   el::Level::Fatal},
                                                                  {"error",   el::Level::Error}
                                                                };

void logging_setup(std::vector<std::string> enabled_log_levels)
{
    el::Configurations log_config;
    log_config.setToDefault();
    log_config.setGlobally(el::ConfigurationType::Enabled, "false");

    log_config.set(el::Level::Info,    el::ConfigurationType::Format, "[%level] %msg");
    log_config.set(el::Level::Verbose, el::ConfigurationType::Format, "[%level] %msg");
    log_config.set(el::Level::Debug,   el::ConfigurationType::Format, "[%level] %fbase:%line %msg");
    log_config.set(el::Level::Warning, el::ConfigurationType::Format, "[%level] %fbase:%line %msg");
    log_config.set(el::Level::Fatal,   el::ConfigurationType::Format, "[%level] %fbase:%line %msg");
    log_config.set(el::Level::Error,   el::ConfigurationType::Format, "[%level] %fbase:%line %msg");
    for(auto level : enabled_log_levels) {
        log_config.set(loglevel_map.at(level), el::ConfigurationType::Enabled, "true");
    }
    el::Loggers::reconfigureLogger("default", log_config);
}
