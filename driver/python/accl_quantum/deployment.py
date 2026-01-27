"""
ACCL-Q Multi-Board RFSoC Deployment Configuration

Provides configuration and setup utilities for deploying ACCL-Q
on multi-board RFSoC test environments (4-8 boards).
"""

import json
import socket
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import threading
import logging

from .constants import (
    ACCLConfig,
    ACCLMode,
    SyncMode,
    CLOCK_PERIOD_NS,
    MAX_RANKS,
)

logger = logging.getLogger(__name__)


class BoardType(Enum):
    """Supported RFSoC board types."""
    ZCU111 = "zcu111"           # Xilinx ZCU111 Evaluation Kit
    ZCU216 = "zcu216"           # Xilinx ZCU216 Evaluation Kit
    RFSoC2x2 = "rfsoc2x2"       # Xilinx RFSoC 2x2 MTS
    RFSoC4x2 = "rfsoc4x2"       # Xilinx RFSoC 4x2
    HTGZRF16 = "htg-zrf16"      # HiTech Global ZRF16
    CUSTOM = "custom"           # Custom board configuration


class NetworkTopology(Enum):
    """Network topology configurations."""
    STAR = "star"               # All boards connect to central switch
    RING = "ring"               # Boards connected in a ring
    TREE = "tree"               # Tree topology with root node
    FULL_MESH = "full_mesh"     # Every board connected to every other
    CUSTOM = "custom"           # User-defined topology


class DeploymentState(Enum):
    """Deployment state machine states."""
    UNINITIALIZED = "uninitialized"
    DISCOVERING = "discovering"
    CONFIGURING = "configuring"
    SYNCHRONIZING = "synchronizing"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class BoardConfig:
    """Configuration for a single RFSoC board."""
    rank: int
    hostname: str
    ip_address: str
    mac_address: str
    board_type: BoardType
    aurora_lanes: int = 4
    aurora_rate_gbps: float = 10.0
    fpga_bitstream: str = ""
    firmware_version: str = ""

    # Hardware-specific settings
    dac_channels: int = 8
    adc_channels: int = 8
    clock_source: str = "internal"  # internal, external, recovered
    reference_freq_mhz: float = 245.76

    # Network settings
    aurora_ports: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    management_port: int = 5000
    data_port: int = 5001

    # Status
    is_online: bool = False
    last_heartbeat: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'rank': self.rank,
            'hostname': self.hostname,
            'ip_address': self.ip_address,
            'mac_address': self.mac_address,
            'board_type': self.board_type.value,
            'aurora_lanes': self.aurora_lanes,
            'aurora_rate_gbps': self.aurora_rate_gbps,
            'fpga_bitstream': self.fpga_bitstream,
            'firmware_version': self.firmware_version,
            'dac_channels': self.dac_channels,
            'adc_channels': self.adc_channels,
            'clock_source': self.clock_source,
            'reference_freq_mhz': self.reference_freq_mhz,
            'aurora_ports': self.aurora_ports,
            'management_port': self.management_port,
            'data_port': self.data_port,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BoardConfig":
        """Create from dictionary."""
        data = data.copy()
        data['board_type'] = BoardType(data['board_type'])
        return cls(**data)


@dataclass
class LinkConfig:
    """Configuration for an Aurora link between boards."""
    source_rank: int
    source_port: int
    dest_rank: int
    dest_port: int
    latency_ns: float = 0.0  # Measured link latency
    is_active: bool = False


@dataclass
class DeploymentConfig:
    """Complete deployment configuration."""
    name: str
    description: str = ""
    topology: NetworkTopology = NetworkTopology.TREE
    num_boards: int = 4
    master_rank: int = 0

    # Board configurations
    boards: Dict[int, BoardConfig] = field(default_factory=dict)

    # Link configurations
    links: List[LinkConfig] = field(default_factory=list)

    # Global settings
    mode: ACCLMode = ACCLMode.DETERMINISTIC
    sync_mode: SyncMode = SyncMode.HARDWARE
    global_timeout_us: int = 1000
    heartbeat_interval_ms: int = 100

    # Clock distribution
    clock_master_rank: int = 0
    sync_accuracy_target_ns: float = 1.0

    # Paths
    bitstream_path: str = ""
    firmware_path: str = ""

    def validate(self) -> List[str]:
        """Validate configuration, return list of errors."""
        errors = []

        if self.num_boards < 2:
            errors.append("Minimum 2 boards required")
        if self.num_boards > MAX_RANKS:
            errors.append(f"Maximum {MAX_RANKS} boards supported")

        if self.master_rank >= self.num_boards:
            errors.append(f"Master rank {self.master_rank} >= num_boards {self.num_boards}")

        if len(self.boards) != self.num_boards:
            errors.append(f"Expected {self.num_boards} board configs, got {len(self.boards)}")

        # Check all ranks are present
        expected_ranks = set(range(self.num_boards))
        actual_ranks = set(self.boards.keys())
        if expected_ranks != actual_ranks:
            missing = expected_ranks - actual_ranks
            extra = actual_ranks - expected_ranks
            if missing:
                errors.append(f"Missing board configs for ranks: {missing}")
            if extra:
                errors.append(f"Extra board configs for ranks: {extra}")

        # Validate topology has sufficient links
        min_links = self._min_links_for_topology()
        if len(self.links) < min_links:
            errors.append(f"Topology {self.topology.value} requires at least {min_links} links")

        return errors

    def _min_links_for_topology(self) -> int:
        """Get minimum links required for topology."""
        n = self.num_boards
        if self.topology == NetworkTopology.STAR:
            return n - 1  # All connect to center
        elif self.topology == NetworkTopology.RING:
            return n  # Each board connects to next
        elif self.topology == NetworkTopology.TREE:
            return n - 1  # N-1 edges in tree
        elif self.topology == NetworkTopology.FULL_MESH:
            return n * (n - 1) // 2  # Complete graph
        return 0

    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        data = {
            'name': self.name,
            'description': self.description,
            'topology': self.topology.value,
            'num_boards': self.num_boards,
            'master_rank': self.master_rank,
            'boards': {str(k): v.to_dict() for k, v in self.boards.items()},
            'links': [
                {
                    'source_rank': l.source_rank,
                    'source_port': l.source_port,
                    'dest_rank': l.dest_rank,
                    'dest_port': l.dest_port,
                }
                for l in self.links
            ],
            'mode': self.mode.value,
            'sync_mode': self.sync_mode.value,
            'global_timeout_us': self.global_timeout_us,
            'heartbeat_interval_ms': self.heartbeat_interval_ms,
            'clock_master_rank': self.clock_master_rank,
            'sync_accuracy_target_ns': self.sync_accuracy_target_ns,
            'bitstream_path': self.bitstream_path,
            'firmware_path': self.firmware_path,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "DeploymentConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        config = cls(
            name=data['name'],
            description=data.get('description', ''),
            topology=NetworkTopology(data['topology']),
            num_boards=data['num_boards'],
            master_rank=data['master_rank'],
            mode=ACCLMode(data['mode']),
            sync_mode=SyncMode(data['sync_mode']),
            global_timeout_us=data['global_timeout_us'],
            heartbeat_interval_ms=data['heartbeat_interval_ms'],
            clock_master_rank=data['clock_master_rank'],
            sync_accuracy_target_ns=data['sync_accuracy_target_ns'],
            bitstream_path=data.get('bitstream_path', ''),
            firmware_path=data.get('firmware_path', ''),
        )

        for rank_str, board_data in data['boards'].items():
            config.boards[int(rank_str)] = BoardConfig.from_dict(board_data)

        for link_data in data['links']:
            config.links.append(LinkConfig(**link_data))

        return config


class BoardDiscovery:
    """
    Discovers and enumerates RFSoC boards on the network.

    Uses multicast UDP for board discovery and management
    protocol for detailed enumeration.
    """

    DISCOVERY_PORT = 5099
    DISCOVERY_MULTICAST = "239.255.0.1"
    DISCOVERY_MAGIC = b"ACCLQ_DISC"

    def __init__(self, timeout_s: float = 5.0):
        self.timeout_s = timeout_s
        self._discovered_boards: Dict[str, BoardConfig] = {}

    def discover(self, expected_boards: int = 0) -> List[BoardConfig]:
        """
        Discover boards on the network.

        Args:
            expected_boards: If > 0, wait until this many boards found

        Returns:
            List of discovered board configurations
        """
        self._discovered_boards.clear()

        # Create multicast socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1.0)

        try:
            # Bind to discovery port
            sock.bind(('', self.DISCOVERY_PORT))

            # Join multicast group
            mreq = struct.pack("4sl",
                socket.inet_aton(self.DISCOVERY_MULTICAST),
                socket.INADDR_ANY)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

            # Send discovery request
            request = self.DISCOVERY_MAGIC + b"\x01"  # Version 1
            sock.sendto(request, (self.DISCOVERY_MULTICAST, self.DISCOVERY_PORT))

            # Collect responses
            start_time = time.time()
            while time.time() - start_time < self.timeout_s:
                try:
                    data, addr = sock.recvfrom(1024)
                    if data.startswith(self.DISCOVERY_MAGIC):
                        board = self._parse_discovery_response(data, addr)
                        if board:
                            self._discovered_boards[addr[0]] = board

                    # Check if we have enough boards
                    if expected_boards > 0 and len(self._discovered_boards) >= expected_boards:
                        break

                except socket.timeout:
                    continue

        finally:
            sock.close()

        return list(self._discovered_boards.values())

    def _parse_discovery_response(self, data: bytes, addr: Tuple[str, int]) -> Optional[BoardConfig]:
        """Parse discovery response packet."""
        try:
            # Skip magic bytes
            data = data[len(self.DISCOVERY_MAGIC):]

            # Parse response (simplified format)
            # Real implementation would have proper TLV encoding
            if len(data) < 20:
                return None

            version = data[0]
            board_type_id = data[1]
            hostname_len = data[2]
            hostname = data[3:3+hostname_len].decode('utf-8')

            # Map board type ID to enum
            board_type_map = {
                0: BoardType.ZCU111,
                1: BoardType.ZCU216,
                2: BoardType.RFSoC2x2,
                3: BoardType.RFSoC4x2,
                4: BoardType.HTGZRF16,
            }
            board_type = board_type_map.get(board_type_id, BoardType.CUSTOM)

            return BoardConfig(
                rank=-1,  # Assigned later
                hostname=hostname,
                ip_address=addr[0],
                mac_address="",  # Would be in response
                board_type=board_type,
                is_online=True,
                last_heartbeat=time.time(),
            )

        except Exception as e:
            logger.warning(f"Failed to parse discovery response: {e}")
            return None

    def probe_board(self, ip_address: str, port: int = 5000) -> Optional[BoardConfig]:
        """
        Probe a specific board for detailed information.

        Args:
            ip_address: Board IP address
            port: Management port

        Returns:
            BoardConfig if successful, None otherwise
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect((ip_address, port))

            # Send probe request
            sock.send(b"ACCLQ_PROBE\x01")

            # Receive response
            response = sock.recv(4096)

            sock.close()

            # Parse probe response (JSON format)
            if response:
                data = json.loads(response.decode('utf-8'))
                return BoardConfig(
                    rank=-1,
                    hostname=data.get('hostname', ''),
                    ip_address=ip_address,
                    mac_address=data.get('mac_address', ''),
                    board_type=BoardType(data.get('board_type', 'custom')),
                    aurora_lanes=data.get('aurora_lanes', 4),
                    aurora_rate_gbps=data.get('aurora_rate_gbps', 10.0),
                    fpga_bitstream=data.get('fpga_bitstream', ''),
                    firmware_version=data.get('firmware_version', ''),
                    dac_channels=data.get('dac_channels', 8),
                    adc_channels=data.get('adc_channels', 8),
                    is_online=True,
                    last_heartbeat=time.time(),
                )

        except Exception as e:
            logger.warning(f"Failed to probe board at {ip_address}: {e}")

        return None


class TopologyBuilder:
    """Builds network topology configurations."""

    @staticmethod
    def build_star(boards: List[BoardConfig], center_rank: int = 0) -> List[LinkConfig]:
        """
        Build star topology with center node.

        All boards connect to the center node.
        """
        links = []
        for board in boards:
            if board.rank != center_rank:
                # Bidirectional link
                links.append(LinkConfig(
                    source_rank=center_rank,
                    source_port=board.rank % 4,  # Distribute across ports
                    dest_rank=board.rank,
                    dest_port=0,
                ))
                links.append(LinkConfig(
                    source_rank=board.rank,
                    source_port=0,
                    dest_rank=center_rank,
                    dest_port=board.rank % 4,
                ))
        return links

    @staticmethod
    def build_ring(boards: List[BoardConfig]) -> List[LinkConfig]:
        """
        Build ring topology.

        Each board connects to the next in sequence.
        """
        links = []
        n = len(boards)
        ranks = sorted([b.rank for b in boards])

        for i, rank in enumerate(ranks):
            next_rank = ranks[(i + 1) % n]
            links.append(LinkConfig(
                source_rank=rank,
                source_port=0,
                dest_rank=next_rank,
                dest_port=1,
            ))
        return links

    @staticmethod
    def build_tree(boards: List[BoardConfig], root_rank: int = 0,
                   fanout: int = 4) -> List[LinkConfig]:
        """
        Build tree topology with specified fanout.

        Optimal for collective operations.
        """
        links = []
        ranks = sorted([b.rank for b in boards])
        n = len(ranks)

        # BFS to assign tree structure
        # Each node has up to 'fanout' children
        for i, rank in enumerate(ranks):
            if rank == root_rank:
                continue

            # Find parent
            parent_idx = (i - 1) // fanout
            parent_rank = ranks[parent_idx]
            child_port = (i - 1) % fanout

            # Bidirectional link
            links.append(LinkConfig(
                source_rank=parent_rank,
                source_port=child_port,
                dest_rank=rank,
                dest_port=0,  # Port 0 is always "up" to parent
            ))
            links.append(LinkConfig(
                source_rank=rank,
                source_port=0,
                dest_rank=parent_rank,
                dest_port=child_port,
            ))

        return links

    @staticmethod
    def build_full_mesh(boards: List[BoardConfig]) -> List[LinkConfig]:
        """
        Build full mesh topology.

        Every board connected to every other board.
        Requires sufficient Aurora ports.
        """
        links = []
        ranks = sorted([b.rank for b in boards])
        n = len(ranks)

        port_counter = {}  # Track port usage per board
        for rank in ranks:
            port_counter[rank] = 0

        for i, src in enumerate(ranks):
            for dst in ranks[i+1:]:
                src_port = port_counter[src]
                dst_port = port_counter[dst]

                links.append(LinkConfig(
                    source_rank=src,
                    source_port=src_port,
                    dest_rank=dst,
                    dest_port=dst_port,
                ))
                links.append(LinkConfig(
                    source_rank=dst,
                    source_port=dst_port,
                    dest_rank=src,
                    dest_port=src_port,
                ))

                port_counter[src] += 1
                port_counter[dst] += 1

        return links


class DeploymentManager:
    """
    Manages ACCL-Q deployment across multiple RFSoC boards.

    Handles:
    - Board discovery and enumeration
    - Configuration distribution
    - FPGA bitstream loading
    - Clock synchronization initialization
    - Health monitoring
    """

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.state = DeploymentState.UNINITIALIZED

        self._discovery = BoardDiscovery()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

        # Callbacks
        self._state_callbacks: List[Callable[[DeploymentState], None]] = []
        self._error_callbacks: List[Callable[[str], None]] = []

    def add_state_callback(self, callback: Callable[[DeploymentState], None]) -> None:
        """Register callback for state changes."""
        self._state_callbacks.append(callback)

    def add_error_callback(self, callback: Callable[[str], None]) -> None:
        """Register callback for errors."""
        self._error_callbacks.append(callback)

    def _set_state(self, state: DeploymentState) -> None:
        """Update state and notify callbacks."""
        self.state = state
        for callback in self._state_callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"State callback error: {e}")

    def _report_error(self, message: str) -> None:
        """Report error to callbacks."""
        logger.error(message)
        for callback in self._error_callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Error callback error: {e}")

    def discover_boards(self) -> List[BoardConfig]:
        """
        Discover boards on network and update configuration.

        Returns:
            List of discovered boards
        """
        self._set_state(DeploymentState.DISCOVERING)

        boards = self._discovery.discover(expected_boards=self.config.num_boards)

        if len(boards) < self.config.num_boards:
            self._report_error(
                f"Found {len(boards)} boards, expected {self.config.num_boards}"
            )
            self._set_state(DeploymentState.ERROR)
            return boards

        # Assign ranks to discovered boards
        for i, board in enumerate(boards[:self.config.num_boards]):
            board.rank = i
            self.config.boards[i] = board

        logger.info(f"Discovered {len(boards)} boards")
        return boards

    def configure_boards(self) -> bool:
        """
        Send configuration to all boards.

        Returns:
            True if all boards configured successfully
        """
        self._set_state(DeploymentState.CONFIGURING)

        success = True
        for rank, board in self.config.boards.items():
            if not self._configure_board(board):
                self._report_error(f"Failed to configure board {rank} ({board.hostname})")
                success = False

        if not success:
            self._set_state(DeploymentState.ERROR)

        return success

    def _configure_board(self, board: BoardConfig) -> bool:
        """Configure a single board."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((board.ip_address, board.management_port))

            # Build configuration message
            config_data = {
                'command': 'configure',
                'rank': board.rank,
                'num_ranks': self.config.num_boards,
                'mode': self.config.mode.value,
                'sync_mode': self.config.sync_mode.value,
                'master_rank': self.config.master_rank,
                'clock_master_rank': self.config.clock_master_rank,
                'timeout_us': self.config.global_timeout_us,
            }

            # Add link configuration for this board
            board_links = [
                {'port': l.source_port, 'dest_rank': l.dest_rank}
                for l in self.config.links
                if l.source_rank == board.rank
            ]
            config_data['links'] = board_links

            # Send configuration
            sock.send(json.dumps(config_data).encode('utf-8'))

            # Wait for acknowledgment
            response = sock.recv(1024)
            sock.close()

            return response == b"OK"

        except Exception as e:
            logger.error(f"Configuration error for {board.hostname}: {e}")
            return False

    def load_bitstreams(self) -> bool:
        """
        Load FPGA bitstreams to all boards.

        Returns:
            True if all bitstreams loaded successfully
        """
        if not self.config.bitstream_path:
            logger.warning("No bitstream path configured, skipping load")
            return True

        success = True
        for rank, board in self.config.boards.items():
            if not self._load_bitstream(board):
                self._report_error(f"Failed to load bitstream on board {rank}")
                success = False

        return success

    def _load_bitstream(self, board: BoardConfig) -> bool:
        """Load bitstream to a single board."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(60.0)  # Bitstream load can take time
            sock.connect((board.ip_address, board.management_port))

            # Send load command
            command = {
                'command': 'load_bitstream',
                'path': board.fpga_bitstream or self.config.bitstream_path,
            }
            sock.send(json.dumps(command).encode('utf-8'))

            # Wait for completion
            response = sock.recv(1024)
            sock.close()

            return response == b"OK"

        except Exception as e:
            logger.error(f"Bitstream load error for {board.hostname}: {e}")
            return False

    def synchronize_clocks(self) -> bool:
        """
        Initialize clock synchronization across all boards.

        Returns:
            True if synchronization successful
        """
        self._set_state(DeploymentState.SYNCHRONIZING)

        try:
            # Step 1: Configure clock master
            master_board = self.config.boards[self.config.clock_master_rank]
            if not self._init_clock_master(master_board):
                self._set_state(DeploymentState.ERROR)
                return False

            # Step 2: Synchronize each slave
            for rank, board in self.config.boards.items():
                if rank != self.config.clock_master_rank:
                    if not self._sync_clock_slave(board):
                        self._set_state(DeploymentState.ERROR)
                        return False

            # Step 3: Verify synchronization accuracy
            max_error = self._measure_sync_accuracy()
            if max_error > self.config.sync_accuracy_target_ns:
                self._report_error(
                    f"Sync accuracy {max_error:.2f}ns exceeds target "
                    f"{self.config.sync_accuracy_target_ns}ns"
                )
                self._set_state(DeploymentState.ERROR)
                return False

            logger.info(f"Clock sync complete, max error: {max_error:.2f}ns")
            return True

        except Exception as e:
            self._report_error(f"Clock synchronization failed: {e}")
            self._set_state(DeploymentState.ERROR)
            return False

    def _init_clock_master(self, board: BoardConfig) -> bool:
        """Initialize clock master board."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((board.ip_address, board.management_port))

            command = {
                'command': 'init_clock_master',
                'reference_freq_mhz': board.reference_freq_mhz,
            }
            sock.send(json.dumps(command).encode('utf-8'))

            response = sock.recv(1024)
            sock.close()

            return response == b"OK"

        except Exception as e:
            logger.error(f"Clock master init error: {e}")
            return False

    def _sync_clock_slave(self, board: BoardConfig) -> bool:
        """Synchronize a slave board's clock."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10.0)
            sock.connect((board.ip_address, board.management_port))

            command = {
                'command': 'sync_clock',
                'master_rank': self.config.clock_master_rank,
                'master_ip': self.config.boards[self.config.clock_master_rank].ip_address,
            }
            sock.send(json.dumps(command).encode('utf-8'))

            response = sock.recv(1024)
            sock.close()

            return response == b"OK"

        except Exception as e:
            logger.error(f"Clock slave sync error for {board.hostname}: {e}")
            return False

    def _measure_sync_accuracy(self) -> float:
        """Measure clock synchronization accuracy across all boards."""
        max_error = 0.0

        for rank, board in self.config.boards.items():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                sock.connect((board.ip_address, board.management_port))

                command = {'command': 'get_sync_error'}
                sock.send(json.dumps(command).encode('utf-8'))

                response = sock.recv(1024)
                sock.close()

                data = json.loads(response.decode('utf-8'))
                error = abs(data.get('phase_error_ns', 0.0))
                max_error = max(max_error, error)

            except Exception as e:
                logger.warning(f"Could not measure sync error for rank {rank}: {e}")

        return max_error

    def deploy(self) -> bool:
        """
        Execute full deployment sequence.

        Returns:
            True if deployment successful
        """
        logger.info(f"Starting deployment: {self.config.name}")

        # Validate configuration
        errors = self.config.validate()
        if errors:
            for error in errors:
                self._report_error(f"Config error: {error}")
            self._set_state(DeploymentState.ERROR)
            return False

        # Discovery (if boards not pre-configured)
        if not self.config.boards:
            boards = self.discover_boards()
            if len(boards) < self.config.num_boards:
                return False

        # Load bitstreams
        if not self.load_bitstreams():
            return False

        # Configure boards
        if not self.configure_boards():
            return False

        # Synchronize clocks
        if not self.synchronize_clocks():
            return False

        # Start health monitoring
        self._start_heartbeat_monitor()

        self._set_state(DeploymentState.READY)
        logger.info("Deployment complete, system ready")
        return True

    def _start_heartbeat_monitor(self) -> None:
        """Start background heartbeat monitoring thread."""
        self._shutdown_event.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        """Background thread for monitoring board health."""
        while not self._shutdown_event.is_set():
            for rank, board in self.config.boards.items():
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1.0)
                    sock.connect((board.ip_address, board.management_port))
                    sock.send(b'{"command": "heartbeat"}')
                    response = sock.recv(64)
                    sock.close()

                    if response == b"OK":
                        board.is_online = True
                        board.last_heartbeat = time.time()
                    else:
                        board.is_online = False

                except Exception:
                    board.is_online = False

            self._shutdown_event.wait(self.config.heartbeat_interval_ms / 1000.0)

    def shutdown(self) -> None:
        """Shutdown deployment and cleanup resources."""
        self._set_state(DeploymentState.SHUTDOWN)
        self._shutdown_event.set()

        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2.0)

        # Send shutdown command to all boards
        for rank, board in self.config.boards.items():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2.0)
                sock.connect((board.ip_address, board.management_port))
                sock.send(b'{"command": "shutdown"}')
                sock.close()
            except Exception:
                pass

        logger.info("Deployment shutdown complete")

    def get_status(self) -> dict:
        """Get deployment status summary."""
        online_boards = sum(1 for b in self.config.boards.values() if b.is_online)

        return {
            'state': self.state.value,
            'name': self.config.name,
            'topology': self.config.topology.value,
            'num_boards': self.config.num_boards,
            'online_boards': online_boards,
            'master_rank': self.config.master_rank,
            'boards': {
                rank: {
                    'hostname': b.hostname,
                    'ip': b.ip_address,
                    'online': b.is_online,
                    'board_type': b.board_type.value,
                }
                for rank, b in self.config.boards.items()
            }
        }


def create_default_deployment(num_boards: int = 4,
                              name: str = "accl-q-test") -> DeploymentConfig:
    """
    Create a default deployment configuration for testing.

    Args:
        num_boards: Number of boards (4-8 typical)
        name: Deployment name

    Returns:
        DeploymentConfig with reasonable defaults
    """
    config = DeploymentConfig(
        name=name,
        description=f"Default {num_boards}-board ACCL-Q deployment",
        topology=NetworkTopology.TREE,
        num_boards=num_boards,
        master_rank=0,
        mode=ACCLMode.DETERMINISTIC,
        sync_mode=SyncMode.HARDWARE,
        clock_master_rank=0,
        sync_accuracy_target_ns=1.0,
    )

    # Create placeholder board configs
    for i in range(num_boards):
        config.boards[i] = BoardConfig(
            rank=i,
            hostname=f"rfsoc-{i}",
            ip_address=f"192.168.1.{100 + i}",
            mac_address=f"00:0a:35:00:00:{i:02x}",
            board_type=BoardType.ZCU216,
        )

    # Build tree topology links
    config.links = TopologyBuilder.build_tree(
        list(config.boards.values()),
        root_rank=0,
        fanout=4
    )

    return config
