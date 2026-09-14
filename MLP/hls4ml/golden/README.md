# Pacote golden HLS4ML para a ZCU104

Estes 30 vetores vêm diretamente das transações aprovadas no RTL co-sim do IP, e são a referência funcional da placa.

- Tipo: `ap_fixed<16,6>` (`AP_TRN`, `AP_WRAP`), 10 bits fracionários.
- Entrada empacotada: `x0[15:0]`, `x1[31:16]`, `x2[47:32]`, `x3[63:48]`.
- Saída: três valores crus signed-16; valor real = `raw / 1024`.
- Resultado esperado: 29/30 corretas e classes exatamente iguais a `predictions`.
- `features_packed_u64` deve ser escrito no registrador/stream de entrada sem nova quantização.

Na placa, teste primeiro todos os vetores deste pacote e só libere o benchmark se a concordância for 30/30 com `predictions`. O único erro esperado de classificação é a amostra cujo rótulo difere da predição; não ajuste o resultado para escondê-lo.
