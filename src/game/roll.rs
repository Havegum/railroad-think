use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Copy, Eq)]
pub struct Roll(pub [u8; 4]);

impl Roll {
    #[must_use]
    pub fn new(roll: [u8; 4]) -> Self {
        Self(roll)
    }

    #[must_use]
    pub fn to_vec(&self) -> Vec<u8> {
        self.0.to_vec()
    }
}

impl From<&Roll> for u32 {
    fn from(roll: &Roll) -> Self {
        let roll = roll.0;
        let a = Self::from(roll[0]);
        let b = Self::from(roll[1]) << 8;
        let c = Self::from(roll[2]) << 16;
        let d = Self::from(roll[3]) << 24;
        a ^ b ^ c ^ d
    }
}

impl PartialEq for Roll {
    fn eq(&self, rhs: &Self) -> bool {
        for i in 0..4 {
            if self.0[i] != rhs.0[i] {
                return false;
            }
        }
        true
    }
}

impl Hash for Roll {
    fn hash<H: Hasher>(&self, state: &mut H) {
        u32::from(self).hash(state);
    }
}

impl std::fmt::Display for Roll {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.0
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join(",")
        )
    }
}
