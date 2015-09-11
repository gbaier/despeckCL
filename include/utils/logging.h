#ifndef LOGGING_H
#define LOGGING_H

#include "easylogging++.h"

#include <map>
#include <string>

void logging_setup(std::vector<std::string> enabled_log_levels);

const std::map<const std::string, const el::Level> loglevel_map {
                                                                  {"debug",   el::Level::Debug},
                                                                  {"info",    el::Level::Info},
                                                                  {"verbose", el::Level::Verbose},
                                                                  {"warning", el::Level::Warning},
                                                                  {"fatal",   el::Level::Fatal},
                                                                  {"error",   el::Level::Error}
                                                                };

#endif
