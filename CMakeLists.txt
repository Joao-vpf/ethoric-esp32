cmake_minimum_required(VERSION 3.16)

# Load variables (IDF_TARGET, PORT, WIFI_SSID, ...) from an optional .env file (see .env.example).
if(EXISTS "${CMAKE_SOURCE_DIR}/.env")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/.env")
    file(READ "${CMAKE_SOURCE_DIR}/.env" ENV_CONTENT)
    string(REPLACE ";" "<SEMICOLON>" ENV_CONTENT "${ENV_CONTENT}")   # keep ';' inside values (e.g. passwords)
    string(REPLACE "\r\n" "\n" ENV_CONTENT "${ENV_CONTENT}")
    string(REPLACE "\r" "\n" ENV_CONTENT "${ENV_CONTENT}")
    string(REGEX REPLACE "\n$" "" ENV_CONTENT "${ENV_CONTENT}")
    string(REPLACE "\n" ";" ENV_LINES "${ENV_CONTENT}")

    foreach(line IN LISTS ENV_LINES)
        string(REGEX MATCH "^[A-Za-z_][A-Za-z0-9_]*=" IS_VAR "${line}")
        if(IS_VAR)
            string(REGEX MATCH "^[A-Za-z_][A-Za-z0-9_]*" VAR_NAME "${line}")
            string(REGEX REPLACE "^[A-Za-z_][A-Za-z0-9_]*=(.*)" "\\1" VAR_VALUE "${line}")
            string(REGEX REPLACE "^\"(.*)\"$" "\\1" VAR_VALUE "${VAR_VALUE}")   # accept WIFI_SSID="my net"
            string(REPLACE "<SEMICOLON>" ";" VAR_VALUE "${VAR_VALUE}")
            set(ENV{${VAR_NAME}} "${VAR_VALUE}")
        endif()
    endforeach()
endif()

# Build only "main" and the components it requires (components/ethoric is found automatically).
set(COMPONENTS main)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

project(ethoric)
