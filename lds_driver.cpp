#include "lds_driver.h"
#include <boost/asio.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <nlohmann/json.hpp> // JSON 库

using json = nlohmann::json;

namespace lds {

LFCDLaser::LFCDLaser(const std::string& port, uint32_t baud_rate, boost::asio::io_service& io)
    : port_(port), baud_rate_(baud_rate), shutting_down_(false), serial_(io, port_) {
    serial_.set_option(boost::asio::serial_port_base::baud_rate(baud_rate_));
    boost::asio::write(serial_, boost::asio::buffer("b", 1));
}

LFCDLaser::~LFCDLaser() {
    boost::asio::write(serial_, boost::asio::buffer("e", 1));
}

void LFCDLaser::poll() {
    uint8_t start_count = 0;
    bool got_scan = false;
    boost::array<uint8_t, 2520> raw_bytes;
    uint8_t good_sets = 0;
    uint32_t motor_speed = 0;
    int index;
    std::vector<double> ranges;
    std::vector<double> angles;

    while (!shutting_down_ && !got_scan) {
        boost::asio::read(serial_, boost::asio::buffer(&raw_bytes[start_count], 1));
        if (start_count == 0) {
            if (raw_bytes[start_count] == 0xFA) {
                start_count = 1;
            }
        } else if (start_count == 1) {
            if (raw_bytes[start_count] == 0xA0) {
                start_count = 0;
                got_scan = true;
                boost::asio::read(serial_, boost::asio::buffer(&raw_bytes[2], 2518));
                ranges.clear();
                angles.clear();
                for (uint16_t i = 0; i < raw_bytes.size(); i += 42) {
                    if (raw_bytes[i] == 0xFA && raw_bytes[i + 1] == (0xA0 + i / 42)) {
                        good_sets++;
                        motor_speed += (raw_bytes[i + 3] << 8) + raw_bytes[i + 2];
                        for (uint16_t j = i + 4; j < i + 40; j += 6) {
                            index = 6 * (i / 42) + (j - 4 - i) / 6;
                            uint8_t byte2 = raw_bytes[j + 2];
                            uint8_t byte3 = raw_bytes[j + 3];
                            uint16_t range = (byte3 << 8) + byte2;
                            double angle = (359 - index) * (M_PI / 180.0);
                            ranges.push_back(range / 1000.0);
                            angles.push_back(angle);
                        }
                    }
                }
                // 将数据转换为 JSON 并发送到标准输出
                json output = {{"ranges", ranges}, {"angles", angles}};
                std::cout << output.dump() << std::endl;
            } else {
                start_count = 0;
            }
        }
    }
}

} // namespace lds

int main(int argc, char** argv) {
    std::string port = "/dev/ttyUSB0";
    int baud_rate = 230400;
    boost::asio::io_service io;

    lds::LFCDLaser laser(port, baud_rate, io);

    // 使用管道与 Python 进行交互
    while (true) {
        laser.poll();
    }
    return 0;
}
