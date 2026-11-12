// TESTING / DESIGN ONLY
//! C9: time. Diary times are local wall-clock times ("2026-10-28T14:03:05"); here they become
//! seconds on one line (as if UTC, which is fine because they are only compared with each other),
//! dates in words, a "now" line for her context, and how long ago something was.

/// Days since 1970-01-01 of a civil date (H. Hinnant's method).
pub fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The civil date of a day number.
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + if m <= 2 { 1 } else { 0 }, m, d)
}

/// "YYYY-MM-DDTHH:MM:SS" (or with a space instead of T) to seconds.
pub fn parse_time(s: &str) -> Option<i64> {
    let s = s.trim();
    if s.len() < 19 {
        return None;
    }
    let n = |a: usize, b: usize| s.get(a..b)?.parse::<i64>().ok();
    let (y, mo, d, h, mi, se) = (n(0, 4)?, n(5, 7)?, n(8, 10)?, n(11, 13)?, n(14, 16)?, n(17, 19)?);
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) || h > 23 || mi > 59 || se > 60 {
        return None;
    }
    Some(days_from_civil(y, mo as u32, d as u32) * 86_400 + h * 3600 + mi * 60 + se)
}

pub fn format_time(t: i64) -> String {
    let (y, m, d) = civil_from_days(t.div_euclid(86_400));
    let s = t.rem_euclid(86_400);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}", s / 3600, s % 3600 / 60, s % 60)
}

#[cfg(windows)]
#[repr(C)]
#[derive(Default)]
struct WinSystemTime {
    year: u16,
    month: u16,
    day_of_week: u16,
    day: u16,
    hour: u16,
    minute: u16,
    second: u16,
    ms: u16,
}

#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn GetLocalTime(st: *mut WinSystemTime);
}

/// The local wall-clock time now, on the same line as `parse_time` (the diary's times are local).
/// Windows: GetLocalTime (as the chat's diary uses); elsewhere UTC.
pub fn local_now() -> i64 {
    #[cfg(windows)]
    {
        let mut st = WinSystemTime::default();
        // SAFETY: GetLocalTime fills the SYSTEMTIME it is given (8 u16 fields, laid out as WinSystemTime).
        unsafe { GetLocalTime(&mut st) };
        days_from_civil(st.year as i64, st.month as u32, st.day as u32) * 86_400 + st.hour as i64 * 3600 + st.minute as i64 * 60 + st.second as i64
    }
    #[cfg(not(windows))]
    {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
    }
}

const DAYS: [&str; 7] = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
const MONTHS: [&str; 12] =
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

pub fn weekday(t: i64) -> &'static str {
    DAYS[(t.div_euclid(86_400) + 4).rem_euclid(7) as usize]
}

/// "Wednesday 28 October 2026".
pub fn date_words(t: i64) -> String {
    let (y, m, d) = civil_from_days(t.div_euclid(86_400));
    format!("{} {d} {} {y}", weekday(t), MONTHS[m as usize - 1])
}

/// The line that tells her what day it is.
pub fn now_line(t: i64) -> String {
    format!("Today is {}.", date_words(t))
}

/// How long before `now` the time `then` was, in plain words.
pub fn ago_words(then: i64, now: i64) -> String {
    let days = now.div_euclid(86_400) - then.div_euclid(86_400);
    let plural = |n: i64, w: &str| if n == 1 { format!("1 {w} ago") } else { format!("{n} {w}s ago") };
    match days {
        i64::MIN..=-1 => "later".into(),
        0 => "today".into(),
        1 => "yesterday".into(),
        2..=13 => plural(days, "day"),
        14..=59 => plural(days / 7, "week"),
        60..=729 => plural(days / 30, "month"),
        _ => plural(days / 365, "year"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_round_trip_and_weekdays_are_right() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        for z in [-800_000, -1, 0, 59, 365, 10_957, 20_754, 2_000_000] {
            let (y, m, d) = civil_from_days(z);
            assert_eq!(days_from_civil(y, m, d), z);
        }
        assert_eq!(weekday(0), "Thursday");
        let t = parse_time("2000-01-01T00:00:00").unwrap();
        assert_eq!(weekday(t), "Saturday");
        let t = parse_time("2026-10-28T14:03:05").unwrap();
        assert_eq!(now_line(t), "Today is Wednesday 28 October 2026.");
        assert_eq!(format_time(t), "2026-10-28T14:03:05");
        assert_eq!(parse_time("2026-10-28 14:03:05"), Some(t));
        assert_eq!(parse_time("2026-13-28T14:03:05"), None);
    }

    #[test]
    fn ago_in_words() {
        let now = parse_time("2026-10-28T09:00:00").unwrap();
        let at = |s: &str| parse_time(s).unwrap();
        assert_eq!(ago_words(at("2026-10-28T08:00:00"), now), "today");
        assert_eq!(ago_words(at("2026-10-27T23:00:00"), now), "yesterday");
        assert_eq!(ago_words(at("2026-10-25T23:00:00"), now), "3 days ago");
        assert_eq!(ago_words(at("2026-10-07T12:00:00"), now), "3 weeks ago");
        assert_eq!(ago_words(at("2026-04-01T12:00:00"), now), "7 months ago");
        assert_eq!(ago_words(at("2023-10-01T12:00:00"), now), "3 years ago");
    }
}
