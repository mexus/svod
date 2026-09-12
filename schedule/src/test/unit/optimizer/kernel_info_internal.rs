use proptest::prelude::*;
use test_case::test_case;

use super::*;
use crate::test::support::prelude::*;

/// `function_name` strips ANSI decoration and hex-escapes anything else that is not
/// a valid identifier character.
#[test_case("test_kernel", "test_kernel"; "already an identifier")]
#[test_case("r_g16l16R32u4", "r_g16l16R32u4"; "kernel name")]
#[test_case("", ""; "empty name")]
#[test_case("\x1b[0m", ""; "a name that is only decoration")]
#[test_case("r\x1b[34mg16\x1b[0m", "rg16"; "colour codes are dropped")]
#[test_case("E_\x1b[31mL?\x1b[0mn6\x1b[K", "E_L3Fn6"; "erase-line code is dropped, question mark escaped")]
#[test_case("a\x1b[31mb\x1b[0mc\x1b[K", "abc"; "several sequences in one name")]
#[test_case("test-kernel+v2", "test2Dkernel2Bv2"; "punctuation is hex escaped")]
#[test_case("é日本", "E965E5672C"; "non-ASCII is hex escaped by code point")]
#[test_case("\x1b[31", "1B5B31"; "an unterminated sequence is escaped, not dropped")]
fn function_name_is_a_valid_identifier(name: &str, expected: &str) {
    assert_eq!(KernelInfo::new(name, vec![], false).function_name(), expected);
}

/// Whatever the name, the function name is a valid generated identifier.
#[test]
fn function_name_is_always_ascii_alphanumeric_or_underscore() {
    proptest!(cheap(), |(name in ".{0,64}")| {
        let function = KernelInfo::new(name.clone(), vec![], false).function_name();
        prop_assert!(
            function.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
            "name {name:?} produced {function:?}"
        );
    });
}
