`timescale 1ns / 1ps

module mlp_iris_axi_wrapper #
(
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 6
)
(
    // ============================================================
    // Clock / Reset
    // ============================================================

    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 s_axi_aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME s_axi_aclk, ASSOCIATED_BUSIF S_AXI, ASSOCIATED_RESET s_axi_aresetn, FREQ_HZ 100000000" *)
    input wire s_axi_aclk,

    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 s_axi_aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME s_axi_aresetn, POLARITY ACTIVE_LOW" *)
    input wire s_axi_aresetn,

    // ============================================================
    // AXI4-Lite Slave
    // ============================================================

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S_AXI, PROTOCOL AXI4LITE, DATA_WIDTH 32, ADDR_WIDTH 6, READ_WRITE_MODE READ_WRITE, FREQ_HZ 100000000" *)
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] s_axi_awaddr,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWPROT" *)
    input  wire [2:0] s_axi_awprot,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *)
    input  wire s_axi_awvalid,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *)
    output wire s_axi_awready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA" *)
    input  wire [C_S_AXI_DATA_WIDTH-1:0] s_axi_wdata,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *)
    input  wire [(C_S_AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID" *)
    input  wire s_axi_wvalid,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY" *)
    output wire s_axi_wready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP" *)
    output reg [1:0] s_axi_bresp,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID" *)
    output reg s_axi_bvalid,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY" *)
    input wire s_axi_bready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *)
    input wire [C_S_AXI_ADDR_WIDTH-1:0] s_axi_araddr,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARPROT" *)
    input wire [2:0] s_axi_arprot,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *)
    input wire s_axi_arvalid,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *)
    output wire s_axi_arready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RDATA" *)
    output reg [C_S_AXI_DATA_WIDTH-1:0] s_axi_rdata,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RRESP" *)
    output reg [1:0] s_axi_rresp,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RVALID" *)
    output reg s_axi_rvalid,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RREADY" *)
    input wire s_axi_rready
);

    // ============================================================
    // Register map
    //
    // 0x00 control/status
    //      write bit 0 = start
    //      write bit 1 = clear done
    //
    //      read bit 1 = done_sticky
    //      read bit 2 = ap_idle
    //      read bit 3 = ap_ready
    //      read bit 4 = busy
    //
    // 0x10 features[31:0]
    // 0x14 features[63:32]
    //
    // 0x20 output 0
    // 0x24 output 1
    // 0x28 output 2
    //
    // 0x2C output valid bits
    // 0x30 cycles of last inference
    // 0x34 invocation counter
    // ============================================================

    reg [63:0] features_reg;

    reg [15:0] output0_reg;
    reg [15:0] output1_reg;
    reg [15:0] output2_reg;

    reg output0_valid;
    reg output1_valid;
    reg output2_valid;

    reg done_sticky;
    reg busy;

    reg [31:0] cycle_counter;
    reg [31:0] last_cycle_count;
    reg [31:0] invocation_count;

    // ============================================================
    // Software command pulses
    // ============================================================

    reg sw_start_pulse;
    reg sw_clear_done_pulse;

    // ============================================================
    // AXI write-channel storage
    // ============================================================

    reg aw_stored;
    reg [C_S_AXI_ADDR_WIDTH-1:0] awaddr_reg;

    reg w_stored;
    reg [C_S_AXI_DATA_WIDTH-1:0] wdata_reg;
    reg [(C_S_AXI_DATA_WIDTH/8)-1:0] wstrb_reg;

    assign s_axi_awready = (!aw_stored) && (!s_axi_bvalid);
    assign s_axi_wready  = (!w_stored)  && (!s_axi_bvalid);

    integer i;

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin

            aw_stored <= 1'b0;
            w_stored  <= 1'b0;

            awaddr_reg <= {C_S_AXI_ADDR_WIDTH{1'b0}};
            wdata_reg  <= {C_S_AXI_DATA_WIDTH{1'b0}};
            wstrb_reg  <= {(C_S_AXI_DATA_WIDTH/8){1'b0}};

            s_axi_bvalid <= 1'b0;
            s_axi_bresp  <= 2'b00;

            features_reg <= 64'd0;

            sw_start_pulse      <= 1'b0;
            sw_clear_done_pulse <= 1'b0;

        end else begin

            sw_start_pulse      <= 1'b0;
            sw_clear_done_pulse <= 1'b0;

            if (s_axi_awready && s_axi_awvalid) begin
                aw_stored  <= 1'b1;
                awaddr_reg <= s_axi_awaddr;
            end

            if (s_axi_wready && s_axi_wvalid) begin
                w_stored <= 1'b1;
                wdata_reg <= s_axi_wdata;
                wstrb_reg <= s_axi_wstrb;
            end

            if (aw_stored && w_stored && !s_axi_bvalid) begin

                aw_stored <= 1'b0;
                w_stored  <= 1'b0;

                s_axi_bvalid <= 1'b1;
                s_axi_bresp  <= 2'b00;

                case (awaddr_reg)

                    6'h00: begin
                        if (wstrb_reg[0]) begin

                            if (wdata_reg[0])
                                sw_start_pulse <= 1'b1;

                            if (wdata_reg[1])
                                sw_clear_done_pulse <= 1'b1;

                        end
                    end

                    6'h10: begin
                        for (i = 0; i < 4; i = i + 1) begin
                            if (wstrb_reg[i])
                                features_reg[(i*8) +: 8]
                                    <= wdata_reg[(i*8) +: 8];
                        end
                    end

                    6'h14: begin
                        for (i = 0; i < 4; i = i + 1) begin
                            if (wstrb_reg[i])
                                features_reg[32 + (i*8) +: 8]
                                    <= wdata_reg[(i*8) +: 8];
                        end
                    end

                    default: begin
                    end

                endcase
            end

            if (s_axi_bvalid && s_axi_bready)
                s_axi_bvalid <= 1'b0;

        end
    end

    // ============================================================
    // AXI read channel
    // ============================================================

    assign s_axi_arready = !s_axi_rvalid;

    always @(posedge s_axi_aclk) begin

        if (!s_axi_aresetn) begin

            s_axi_rvalid <= 1'b0;
            s_axi_rdata  <= 32'd0;
            s_axi_rresp  <= 2'b00;

        end else begin

            if (s_axi_arready && s_axi_arvalid) begin

                s_axi_rvalid <= 1'b1;
                s_axi_rresp  <= 2'b00;

                case (s_axi_araddr)

                    6'h00:
                        s_axi_rdata <= {
                            27'd0,
                            busy,
                            core_ap_ready,
                            core_ap_idle,
                            done_sticky,
                            1'b0
                        };

                    6'h10:
                        s_axi_rdata <= features_reg[31:0];

                    6'h14:
                        s_axi_rdata <= features_reg[63:32];

                    6'h20:
                        s_axi_rdata <= {16'd0, output0_reg};

                    6'h24:
                        s_axi_rdata <= {16'd0, output1_reg};

                    6'h28:
                        s_axi_rdata <= {16'd0, output2_reg};

                    6'h2C:
                        s_axi_rdata <= {
                            29'd0,
                            output2_valid,
                            output1_valid,
                            output0_valid
                        };

                    6'h30:
                        s_axi_rdata <= last_cycle_count;

                    6'h34:
                        s_axi_rdata <= invocation_count;

                    default:
                        s_axi_rdata <= 32'd0;

                endcase
            end

            if (s_axi_rvalid && s_axi_rready)
                s_axi_rvalid <= 1'b0;

        end
    end

    // ============================================================
    // HLS mlp_iris signals
    // ============================================================

    reg core_ap_start;
    reg core_features_ap_vld;

    wire core_ap_done;
    wire core_ap_idle;
    wire core_ap_ready;

    wire [15:0] core_out0;
    wire [15:0] core_out1;
    wire [15:0] core_out2;

    wire core_out0_vld;
    wire core_out1_vld;
    wire core_out2_vld;

    // ============================================================
    // Accelerator control
    // ============================================================

    always @(posedge s_axi_aclk) begin

        if (!s_axi_aresetn) begin

            core_ap_start        <= 1'b0;
            core_features_ap_vld <= 1'b0;

            output0_reg <= 16'd0;
            output1_reg <= 16'd0;
            output2_reg <= 16'd0;

            output0_valid <= 1'b0;
            output1_valid <= 1'b0;
            output2_valid <= 1'b0;

            done_sticky <= 1'b0;
            busy        <= 1'b0;

            cycle_counter   <= 32'd0;
            last_cycle_count <= 32'd0;
            invocation_count <= 32'd0;

        end else begin

            if (sw_clear_done_pulse)
                done_sticky <= 1'b0;

            if (sw_start_pulse && !busy) begin

                core_ap_start        <= 1'b1;
                core_features_ap_vld <= 1'b1;

                busy        <= 1'b1;
                done_sticky <= 1'b0;

                output0_valid <= 1'b0;
                output1_valid <= 1'b0;
                output2_valid <= 1'b0;

                cycle_counter <= 32'd0;

                invocation_count <= invocation_count + 1'b1;

            end else if (busy) begin

                cycle_counter <= cycle_counter + 1'b1;

                // ap_start and input-valid are held until the
                // accelerator acknowledges readiness.
                if (core_ap_ready) begin
                    core_ap_start        <= 1'b0;
                    core_features_ap_vld <= 1'b0;
                end

                if (core_out0_vld) begin
                    output0_reg   <= core_out0;
                    output0_valid <= 1'b1;
                end

                if (core_out1_vld) begin
                    output1_reg   <= core_out1;
                    output1_valid <= 1'b1;
                end

                if (core_out2_vld) begin
                    output2_reg   <= core_out2;
                    output2_valid <= 1'b1;
                end

                if (core_ap_done) begin

                    busy        <= 1'b0;
                    done_sticky <= 1'b1;

                    core_ap_start        <= 1'b0;
                    core_features_ap_vld <= 1'b0;

                    last_cycle_count <= cycle_counter + 1'b1;

                end
            end

        end
    end

    // ============================================================
    // Original hls4ml/Vitis HLS core
    // ============================================================

    mlp_iris_core mlp_core_i
    (
        .ap_clk              (s_axi_aclk),
        .ap_rst              (~s_axi_aresetn),

        .ap_start            (core_ap_start),
        .ap_done             (core_ap_done),
        .ap_idle             (core_ap_idle),
        .ap_ready            (core_ap_ready),

        .features            (features_reg),
        .features_ap_vld     (core_features_ap_vld),

        .layer7_out_0        (core_out0),
        .layer7_out_0_ap_vld (core_out0_vld),

        .layer7_out_1        (core_out1),
        .layer7_out_1_ap_vld (core_out1_vld),

        .layer7_out_2        (core_out2),
        .layer7_out_2_ap_vld (core_out2_vld)
    );

endmodule
