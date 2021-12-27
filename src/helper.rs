// pub mod macros {

//     #[macro_export]
//     macro_rules! assert_delta {
//         ($x:expr, $y:expr, $d:expr) => {
//             if !($x - $y < $d || $y - $x < $d) {
//                 panic!();
//             }
//         };
//     }

//     #[macro_export]
//     macro_rules! assert_delta_arr {
//         ($x:expr, $y: expr, $d: expr) => {
//             assert_eq!($x.len(), $y.len());
//             for i in 0..$x.len() {
//                 assert_delta!($x[i], $y[i], $d);
//             }
//         };
//     }
// }
