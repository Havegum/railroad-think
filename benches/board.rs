use criterion::{black_box, criterion_group, criterion_main, Criterion};
use railroad_ink_solver::board::Board;
use railroad_ink_solver::pieces::get_piece;

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("decode board", |b| b.iter(|| {
        let encoding = black_box(String::from(
            "6F0315F0113G0122G0102F0121F0220F0310B0311B0231C0301D0133A0303B0104B0315B0D06B0315C0305D010",
          ));
        let _board = Board::decode(&encoding);
    }));

    let pieces = [
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
    ];

    c.bench_function("permute pieces", |b| {
        b.iter(|| {
            for piece in pieces.iter().filter_map(|&id| get_piece(id)) {
                black_box(piece).get_permutations();
            }
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
