library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity Accel_dma_wrapper is
    Port (
        ap_clk   : in  std_logic;
        ap_rst_n : in  std_logic;

        s_axis_tdata  : in  std_logic_vector(127 downto 0);
        s_axis_tvalid : in  std_logic;
        s_axis_tready : out std_logic;

        m_axis_tdata  : out std_logic_vector(511 downto 0);
        m_axis_tvalid : out std_logic;
        m_axis_tready : in  std_logic;
        m_axis_tlast  : out std_logic;

        ap_start : out std_logic;
        ap_done  : in  std_logic;
        ap_idle  : in  std_logic;
        ap_ready : in  std_logic;

        accel_input_data  : out std_logic_vector(95 downto 0);
        accel_input_valid : out std_logic;
        accel_input_ready : in  std_logic;

        accel_output_data  : in  std_logic_vector(319 downto 0);
        accel_output_valid : in  std_logic;
        accel_output_ready : out std_logic
    );
end Accel_dma_wrapper;

architecture rtl of Accel_dma_wrapper is
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

begin

    ap_start <= '1';

    -- DMA MM2S 128 bits -> ResNet8 96 bits
    accel_input_data  <= s_axis_tdata(95 downto 0);
    accel_input_valid <= s_axis_tvalid;
    s_axis_tready     <= accel_input_ready;

    -- ResNet8 320 bits -> DMA S2MM 512 bits
    m_axis_tdata(319 downto 0)   <= accel_output_data;
    m_axis_tdata(511 downto 320) <= (others => '0');

    m_axis_tvalid       <= accel_output_valid;
    accel_output_ready  <= m_axis_tready;

    -- Um único beat de saída por inferência
    m_axis_tlast <= accel_output_valid;

end rtl;
