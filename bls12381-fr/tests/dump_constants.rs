use zkhash::ark_ff::{BigInteger, PrimeField};
use zkhash::fields::bls12::FpBLS12;
use zkhash::poseidon2::poseidon2_instance_bls12::RC3;

#[test]
fn dump_bls12381_rc3_constants() {
    for (round, rc) in RC3.iter().enumerate() {
        for (i, val) in rc.iter().enumerate() {
            let bytes = val.into_bigint().to_bytes_be();
            let hex_str: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
            println!("RC[{}][{}] = 0x{}", round, i, hex_str);
        }
    }
}
