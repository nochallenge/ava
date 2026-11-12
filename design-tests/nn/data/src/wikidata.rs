// TESTING / DESIGN ONLY
//! wikidata: the three small Wikidata tables v42 uses (CC0), read from `downloads/wikidata`.
//!
//!   q1.csv  countryLabel,capitalLabel,continentLabel   (a country may have several rows: several
//!                                                       capitals, or several continents)
//!   q2.csv  elementLabel,symbol,number                  (number written like "2.0")
//!   q3.csv  countryLabel,langLabel                      (not used for restatements)
//!
//! CSV: comma separated, a field may be in double quotes (then it may hold commas, and "" is one
//! quote), line ends `\r\n` or `\n`, the first line is a header and is skipped.

use std::collections::BTreeMap;
use std::io;
use std::path::Path;

/// One country: its capitals and continents, each list in file order without repeats.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Country {
    pub name: String,
    pub capitals: Vec<String>,
    pub continents: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Element {
    pub name: String,
    pub symbol: String,
    pub number: u32,
}

#[derive(Clone, Debug, Default)]
pub struct Wikidata {
    /// By country name (sorted, so every use is in a fixed order).
    pub countries: BTreeMap<String, Country>,
    /// In file order.
    pub elements: Vec<Element>,
}

/// Split one CSV line into fields (quotes handled as described above).
pub fn csv_fields(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quoted = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if quoted {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    cur.push('"');
                    chars.next();
                } else {
                    quoted = false;
                }
            } else {
                cur.push(c);
            }
        } else if c == '"' {
            quoted = true;
        } else if c == ',' {
            out.push(std::mem::take(&mut cur));
        } else {
            cur.push(c);
        }
    }
    out.push(cur);
    out
}

fn rows(text: &str) -> impl Iterator<Item = Vec<String>> + '_ {
    text.lines().skip(1).map(|l| l.trim_end_matches('\r')).filter(|l| !l.trim().is_empty()).map(csv_fields)
}

fn push_new(v: &mut Vec<String>, s: &str) {
    let s = s.trim();
    if !s.is_empty() && !v.iter().any(|x| x == s) {
        v.push(s.to_string());
    }
}

impl Wikidata {
    /// Parse the text of q1.csv and q2.csv.
    pub fn parse(q1: &str, q2: &str) -> Result<Wikidata, String> {
        let mut w = Wikidata::default();
        for (i, f) in rows(q1).enumerate() {
            if f.len() < 3 {
                return Err(format!("q1.csv row {}: want 3 fields, got {}", i + 2, f.len()));
            }
            let name = f[0].trim();
            if name.is_empty() {
                continue;
            }
            let c = w.countries.entry(name.to_string()).or_insert_with(|| Country { name: name.to_string(), ..Default::default() });
            push_new(&mut c.capitals, &f[1]);
            push_new(&mut c.continents, &f[2]);
        }
        for (i, f) in rows(q2).enumerate() {
            if f.len() < 3 {
                return Err(format!("q2.csv row {}: want 3 fields, got {}", i + 2, f.len()));
            }
            let number = f[2].trim().parse::<f64>().map_err(|_| format!("q2.csv row {}: bad number {:?}", i + 2, f[2]))?;
            w.elements.push(Element { name: f[0].trim().to_string(), symbol: f[1].trim().to_string(), number: number as u32 });
        }
        Ok(w)
    }

    /// Read `DIR/q1.csv` and `DIR/q2.csv`.
    pub fn load(dir: impl AsRef<Path>) -> io::Result<Wikidata> {
        let d = dir.as_ref();
        let q1 = std::fs::read_to_string(d.join("q1.csv"))?;
        let q2 = std::fs::read_to_string(d.join("q2.csv"))?;
        Wikidata::parse(&q1, &q2).map_err(io::Error::other)
    }

    pub fn element(&self, name: &str) -> Option<&Element> {
        self.elements.iter().find(|e| e.name == name)
    }

    /// Every name in the tables: countries, capitals, continents, elements, symbols (unique, sorted).
    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = Vec::new();
        for c in self.countries.values() {
            v.push(c.name.clone());
            v.extend(c.capitals.iter().cloned());
            v.extend(c.continents.iter().cloned());
        }
        for e in &self.elements {
            v.push(e.name.clone());
            v.push(e.symbol.clone());
        }
        v.sort();
        v.dedup();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_quotes_and_crlf() {
        assert_eq!(csv_fields("a,\"b, c\",d"), vec!["a", "b, c", "d"]);
        assert_eq!(csv_fields("\"say \"\"hi\"\"\",x"), vec!["say \"hi\"", "x"]);
        assert_eq!(csv_fields(""), vec![""]);
    }

    #[test]
    fn parses_both_tables() {
        let q1 = "countryLabel,capitalLabel,continentLabel\r\nUnited States,\"Washington, D.C.\",North America\r\nRussia,Moscow,Europe\r\nRussia,Moscow,Asia\r\nBolivia,Sucre,South America\r\nBolivia,La Paz,South America\r\n";
        let q2 = "elementLabel,symbol,number\nhydrogen,H,1.0\nhelium,He,2.0\n";
        let w = Wikidata::parse(q1, q2).unwrap();
        assert_eq!(w.countries["United States"].capitals, vec!["Washington, D.C."]);
        assert_eq!(w.countries["Russia"].continents, vec!["Europe", "Asia"]);
        assert_eq!(w.countries["Bolivia"].capitals.len(), 2);
        assert_eq!(w.element("helium").unwrap(), &Element { name: "helium".into(), symbol: "He".into(), number: 2 });
        let n = w.names();
        assert!(n.contains(&"Washington, D.C.".to_string()) && n.contains(&"He".to_string()) && n.contains(&"Asia".to_string()));
    }
}
