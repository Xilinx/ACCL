#include "communicator.hpp"
#include "constants.hpp"
#include <sstream>

namespace ACCL {
Communicator::Communicator(CCLO *cclo, const std::vector<rank_t> &ranks,
                           int rank, addr_t communicators_addr)
    : cclo(cclo), _ranks(ranks), _rank(rank),
      _communicators_addr(communicators_addr) {
  addr_t addr = _communicators_addr;
  cclo->write(addr, _ranks.size());
  addr += 4;
  cclo->write(addr, _rank);

  for (auto &r : _ranks) {
    addr += 4;
    cclo->write(addr, ip_encode(r.ip));
    addr += 4;
    cclo->write(addr, r.port);
    addr += 4;
    if (r.session_id < 0) {
      cclo->write(addr, 0xFFFFFFFF);
    } else {
      cclo->write(addr, r.session_id);
    }
    addr += 4;
    cclo->write(addr, r.max_segment_size);
  }
}

std::string Communicator::dump() {
  std::stringstream stream;
  addr_t addr = _communicators_addr;

  stream << "local rank: " << cclo->read(addr);
  addr += 4;
  size_t n = cclo->read(addr);
  stream << " \t number of ranks: " << n << std::endl;
  for (size_t i = 0; i < n; ++i) {
    addr += 4;
    std::string ip_address = ip_decode(cclo->read(addr));
    addr += 4;
    // when using the UDP stack, write the rank number into the port register
    // the actual port is programmed into the stack itself
    val_t port = cclo->read(addr);

    // leave 2 32 bit space for inbound/outbound_seq_number
    addr += 4;
    val_t inbound_seq_number = cclo->read(addr);
    addr += 4;
    val_t outbound_seq_number = cclo->read(addr);
    // a 32 bit integer is dedicated to session id
    addr += 4;
    val_t session = cclo->read(addr);
    addr += 4;
    val_t max_seg_size = cclo->read(addr);
    stream << "> rank " << i << " (ip " << ip_address << ":" << port
           << " ; session " << session << " ; max segment size" << max_seg_size
           << " : <- inbound seq number " << inbound_seq_number
           << ", -> outbound seq number" << outbound_seq_number << std::endl;
  }

  return stream.str();
}
} // namespace ACCL
