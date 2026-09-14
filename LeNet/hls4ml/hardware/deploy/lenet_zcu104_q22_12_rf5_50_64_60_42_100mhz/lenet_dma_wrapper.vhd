
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity lenet_dma_wrapper is
    Port (
        ap_clk   : in  std_logic;
        ap_rst_n : in  std_logic;

        -- AXI4-Stream vindo do DMA MM2S
        s_axis_tdata  : in  std_logic_vector(31 downto 0);
        s_axis_tvalid : in  std_logic;
        s_axis_tready : out std_logic;
        s_axis_tlast  : in  std_logic;

        -- AXI4-Stream enviado ao DMA S2MM
        m_axis_tdata  : out std_logic_vector(511 downto 0);
        m_axis_tvalid : out std_logic;
        m_axis_tready : in  std_logic;
        m_axis_tlast  : out std_logic;

        -- Controle ap_ctrl_hs da LeNet
        accel_ap_start : out std_logic;
        accel_ap_done  : in  std_logic;
        accel_ap_idle  : in  std_logic;
        accel_ap_ready : in  std_logic;

        -- Stream de entrada da LeNet
        accel_input_data  : out std_logic_vector(31 downto 0);
        accel_input_valid : out std_logic;
        accel_input_ready : in  std_logic;

        -- Stream de saida da LeNet
        accel_output_data  : in  std_logic_vector(319 downto 0);
        accel_output_valid : in  std_logic;
        accel_output_ready : out std_logic
    );
end lenet_dma_wrapper;

architecture rtl of lenet_dma_wrapper is

    attribute X_INTERFACE_INFO : string;
    attribute X_INTERFACE_PARAMETER : string;

    attribute X_INTERFACE_INFO of ap_clk : signal is
        "xilinx.com:signal:clock:1.0 ap_clk CLK";

    attribute X_INTERFACE_PARAMETER of ap_clk : signal is
        "ASSOCIATED_BUSIF s_axis:m_axis, ASSOCIATED_RESET ap_rst_n";

    attribute X_INTERFACE_INFO of ap_rst_n : signal is
        "xilinx.com:signal:reset:1.0 ap_rst_n RST";

    attribute X_INTERFACE_PARAMETER of ap_rst_n : signal is
        "POLARITY ACTIVE_LOW";

    attribute X_INTERFACE_INFO of s_axis_tdata : signal is
        "xilinx.com:interface:axis:1.0 s_axis TDATA";

    attribute X_INTERFACE_INFO of s_axis_tvalid : signal is
        "xilinx.com:interface:axis:1.0 s_axis TVALID";

    attribute X_INTERFACE_INFO of s_axis_tready : signal is
        "xilinx.com:interface:axis:1.0 s_axis TREADY";

    attribute X_INTERFACE_INFO of s_axis_tlast : signal is
        "xilinx.com:interface:axis:1.0 s_axis TLAST";

    attribute X_INTERFACE_PARAMETER of s_axis_tdata : signal is
        "TDATA_NUM_BYTES 4, HAS_TKEEP 0, HAS_TSTRB 0, HAS_TLAST 1";

    attribute X_INTERFACE_INFO of m_axis_tdata : signal is
        "xilinx.com:interface:axis:1.0 m_axis TDATA";

    attribute X_INTERFACE_INFO of m_axis_tvalid : signal is
        "xilinx.com:interface:axis:1.0 m_axis TVALID";

    attribute X_INTERFACE_INFO of m_axis_tready : signal is
        "xilinx.com:interface:axis:1.0 m_axis TREADY";

    attribute X_INTERFACE_INFO of m_axis_tlast : signal is
        "xilinx.com:interface:axis:1.0 m_axis TLAST";

    attribute X_INTERFACE_PARAMETER of m_axis_tdata : signal is
        "TDATA_NUM_BYTES 64, HAS_TKEEP 0, HAS_TSTRB 0, HAS_TLAST 1";

begin

    -- LeNet fica permanentemente habilitada.
    accel_ap_start <= '1';

    -- Entrada DMA -> LeNet.
    accel_input_data  <= s_axis_tdata;
    accel_input_valid <= s_axis_tvalid;
    s_axis_tready     <= accel_input_ready;

    -- O TLAST de entrada e usado pelo DMA para delimitar o pacote.
    -- A LeNet consome exatamente 784 handshakes e nao possui TLAST.

    -- Saida LeNet 320 bits -> DMA 512 bits.
    m_axis_tdata(319 downto 0)   <= accel_output_data;
    m_axis_tdata(511 downto 320) <= (others => '0');

    m_axis_tvalid      <= accel_output_valid;
    accel_output_ready <= m_axis_tready;

    -- Existe somente um beat na saida.
    -- TLAST acompanha TVALID e permanece valido sob backpressure.
    m_axis_tlast <= accel_output_valid;

end rtl;

