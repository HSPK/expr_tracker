pub(crate) use crate::history::{
    encode_browser_value as browser_value, parse_compatible_json as parse,
};

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn legacy_values_are_strict_browser_json_without_replacing_string_contents() {
        let parsed = parse(
            br#"{"v":NaN,"p":Infinity,"n":-Infinity,"s":"NaN Infinity","huge":9007199254740993,"zero":0}"#,
        )
        .unwrap();
        let value = browser_value(parsed.value);
        assert_eq!(value["v"]["$et.nonfinite"], "NaN");
        assert_eq!(value["s"], "NaN Infinity");
        assert_eq!(value["huge"]["$et.integer"], "9007199254740993");
        assert_eq!(value["zero"], 0);
        serde_json::from_str::<Value>(&serde_json::to_string(&value).unwrap()).unwrap();
    }

    #[test]
    fn surrogate_strings_remain_visible_and_valid_pairs_decode_normally() {
        let parsed =
            parse(br#"{"s":"a\ud800b","valid":"\ud83d\ude00","slash":"\\ud800"}"#).unwrap();
        assert!(parsed.escaped_surrogates);
        assert_eq!(parsed.value["s"], "a\\ud800b");
        assert_eq!(parsed.value["valid"], "\u{1f600}");
        assert_eq!(parsed.value["slash"], "\\ud800");
    }

    #[test]
    fn malformed_tokens_are_not_accepted_as_numbers() {
        assert!(parse(br#"{"v":NaNextra}"#).is_err());
        assert!(parse(br#"{"v":unterminated}"#).is_err());
    }
}
