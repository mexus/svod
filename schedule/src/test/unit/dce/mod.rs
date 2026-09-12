//! Dead-code elimination: the folds that delete a branch no condition can take and a loop
//! no trip count can enter. Both run inside the `Matchers::dce` table, so a regression here
//! shows up as dead code surviving into codegen rather than as a wrong value.

pub mod dead_branches;
pub mod dead_loops;
