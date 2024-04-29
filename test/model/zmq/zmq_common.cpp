/*******************************************************************************
#  Copyright (C) 2021 Xilinx, Inc
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
# *******************************************************************************/

#include "zmq_common.h"
#include <string>

Json::Value to_json(zmq::message_t &message) {
    Json::Reader reader;
    Json::Value json;
    reader.parse(message.to_string(), json);
    return json;
}

zmq::message_t to_message(Json::Value &request_json){
    Json::StreamWriterBuilder builder;
    builder["indentation"] = ""; // minimize output
    const std::string message = Json::writeString(builder, request_json);
    return zmq::message_t(message);
}
