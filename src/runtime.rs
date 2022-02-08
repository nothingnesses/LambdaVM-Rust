#![allow(clippy::identity_op)]
#![allow(dead_code)]
#![allow(non_snake_case)]

use std::collections::{hash_map, HashMap};

// Constants
// ---------

const U64_PER_KB: u64 = 0x80;
const U64_PER_MB: u64 = 0x20000;
const U64_PER_GB: u64 = 0x8000000;

pub const MAX_ARITY: u64 = 16;
// Each worker has a fraction of the total.
pub const MEM_SPACE: u64 = U64_PER_GB;
// uses 32 MB, covers heaps up to 2 GB
pub const SEEN_SIZE: usize = 4194304;

// Terms
// -----
// HVM's runtime stores terms in a 64-bit memory. Each element is a Link, which
// usually points to a constructor. It stores a Tag representing the ctor's
// variant, and possibly a position on the memory. So, for example, `Lnk ptr =
// APP * TAG | 137` creates a pointer to an app node stored on position 137.
// Some links deal with variables: DP0, DP1, VAR, ARG and ERA.  The OP2 link
// represents a numeric operation, and U32 and F32 links represent unboxed nums.

pub const VAL: u64 = 1;
pub const EXT: u64 = 0x100000000;
pub const ARI: u64 = 0x100000000000000;
pub const TAG: u64 = 0x1000000000000000;

// points to the dup node that binds this variable (left side)
pub const DP0: u64 = 0x0;
// points to the dup node that binds this variable (right side)
pub const DP1: u64 = 0x1;
// points to the λ that binds this variable
pub const VAR: u64 = 0x2;
// points to the occurrence of a bound variable a linear argument
pub const ARG: u64 = 0x3;
// signals that a binder doesn't use its bound variable
pub const ERA: u64 = 0x4;
// arity = 2
pub const LAM: u64 = 0x5;
// arity = 2
pub const APP: u64 = 0x6;
// arity = 2
pub const SUP: u64 = 0x7;
// arity = user defined
pub const CTR: u64 = 0x8;
// arity = user defined
pub const CAL: u64 = 0x9;
// arity = 2
pub const OP2: u64 = 0xA;
// arity = 0 (unboxed)
pub const U32: u64 = 0xB;
// arity = 0 (unboxed)
pub const F32: u64 = 0xC;
// not used
pub const NIL: u64 = 0xF;

pub const ADD: u64 = 0x0;
pub const SUB: u64 = 0x1;
pub const MUL: u64 = 0x2;
pub const DIV: u64 = 0x3;
pub const MOD: u64 = 0x4;
pub const AND: u64 = 0x5;
pub const OR: u64 = 0x6;
pub const XOR: u64 = 0x7;
pub const SHL: u64 = 0x8;
pub const SHR: u64 = 0x9;
pub const LTN: u64 = 0xA;
pub const LTE: u64 = 0xB;
pub const EQL: u64 = 0xC;
pub const GTE: u64 = 0xD;
pub const GTN: u64 = 0xE;
pub const NEQ: u64 = 0xF;

// Types
// -----

pub type Lnk = u64;

pub type Rewriter = Box<dyn Fn(&mut Worker, u64, Lnk) -> bool>;

pub struct Function {
  pub arity: u64,
  pub stricts: Vec<u64>,
  pub rewriter: Rewriter,
}

pub struct Worker {
  pub node: Vec<Lnk>,
  pub size: u64,
  pub free: Vec<Vec<u64>>,
  pub cost: u64,
}

pub fn new_worker() -> Worker {
  Worker { node: vec![0; 6 * 0x8000000], size: 0, free: vec![vec![]; 16], cost: 0 }
}

// Globals
// -------

static mut SEEN_DATA: [u64; SEEN_SIZE] = [0; SEEN_SIZE];

// Constructors
// ------------
// Creating, storing and reading Lnks, allocating and freeing memory.

pub fn Var(pos: u64) -> Lnk {
  (VAR * TAG) | pos
}

pub fn Dp0(col: u64, pos: u64) -> Lnk {
  (DP0 * TAG) | (col * EXT) | pos
}

pub fn Dp1(col: u64, pos: u64) -> Lnk {
  (DP1 * TAG) | (col * EXT) | pos
}

pub fn Arg(pos: u64) -> Lnk {
  (ARG * TAG) | pos
}

pub fn Era() -> Lnk {
  ERA * TAG
}

pub fn Lam(pos: u64) -> Lnk {
  (LAM * TAG) | pos
}

pub fn App(pos: u64) -> Lnk {
  (APP * TAG) | pos
}

pub fn Sup(col: u64, pos: u64) -> Lnk {
  (SUP * TAG) | (col * EXT) | pos
}

pub fn Op2(ope: u64, pos: u64) -> Lnk {
  (OP2 * TAG) | (ope * EXT) | pos
}

pub fn U_32(val: u64) -> Lnk {
  (U32 * TAG) | val
}

pub fn Nil() -> Lnk {
  NIL * TAG
}

pub fn Ctr(ari: u64, fun: u64, pos: u64) -> Lnk {
  (CTR * TAG) | (ari * ARI) | (fun * EXT) | pos
}

pub fn Cal(ari: u64, fun: u64, pos: u64) -> Lnk {
  (CAL * TAG) | (ari * ARI) | (fun * EXT) | pos
}

// Getters
// -------

pub fn get_tag(lnk: Lnk) -> u64 {
  lnk / TAG
}

pub fn get_ext(lnk: Lnk) -> u64 {
  (lnk / EXT) & 0xFFFFFF
}

pub fn get_val(lnk: Lnk) -> u64 {
  lnk & 0xFFFFFFFF
}

pub fn get_ari(lnk: Lnk) -> u64 {
  (lnk / ARI) & 0xF
}

pub fn get_loc(lnk: Lnk, arg: u64) -> u64 {
  get_val(lnk) + arg
}

// Memory
// ------

// Dereferences a Lnk, getting what is stored on its target position
pub fn ask_lnk(mem: &Worker, loc: u64) -> Lnk {
  unsafe { *mem.node.get_unchecked(loc as usize) }
  // mem.node[loc as usize]
}

// Dereferences the nth argument of the Term represented by this Lnk
pub fn ask_arg(mem: &Worker, term: Lnk, arg: u64) -> Lnk {
  ask_lnk(mem, get_loc(term, arg))
}

// This inserts a value in another. It just writes a position in memory if
// `value` is a constructor. If it is VAR, DP0 or DP1, it also updates the
// corresponding λ or dup binder.
pub fn link(mem: &mut Worker, loc: u64, lnk: Lnk) -> Lnk {
  unsafe {
    // mem.node[loc as usize] = lnk;
    *mem.node.get_unchecked_mut(loc as usize) = lnk;
    match get_tag(lnk) {
      VAR | DP0 | DP1 => {
        // let pos = get_loc(lnk, if get_tag(lnk) == DP1 { 1 } else { 0 });
        let pos = get_loc(lnk, get_tag(lnk) & 0x01);
        // mem.node[pos as usize] = Arg(loc);
        *mem.node.get_unchecked_mut(pos as usize) = Arg(loc);
      }
      _ => {}
    }
  }
  lnk
}

// Allocates a block of memory, up to 16 words long
pub fn alloc(mem: &mut Worker, size: u64) -> u64 {
  if size == 0 {
    0
  } else if let Some(reuse) = mem.free[size as usize].pop() {
    reuse
  } else {
    let loc = mem.size;
    mem.size += size;
    loc
  }
}

// Frees a block of memory by adding its position to a freelist
pub fn clear(mem: &mut Worker, loc: u64, size: u64) {
  mem.free[size as usize].push(loc);
}

// Garbage Collection
// ------------------

// This clears the memory used by a term that becomes unreachable. It just frees
// all its nodes recursivelly. This is called as soon as a term goes out of
// scope. No global GC pass is necessary to find unreachable terms!
// HVM can still produce some garbage in very uncommon situations that are
// mostly irrelevant in practice. Absolute GC-freedom, though, requires
// uncommenting the `reduce` lines below, but this would make HVM not 100% lazy
// in some cases, so it should be called in a separate thread.
pub fn collect(mem: &mut Worker, term: Lnk) {
  match get_tag(term) {
    DP0 => {
      link(mem, get_loc(term, 0), Era());
      // reduce(mem, get_loc(ask_arg(mem,term,1),0));
    }
    DP1 => {
      link(mem, get_loc(term, 1), Era());
      // reduce(mem, get_loc(ask_arg(mem,term,0),0));
    }
    VAR => {
      link(mem, get_loc(term, 0), Era());
    }
    LAM => {
      if get_tag(ask_arg(mem, term, 0)) != ERA {
        link(mem, get_loc(ask_arg(mem, term, 0), 0), Era());
      }
      collect(mem, ask_arg(mem, term, 1));
      clear(mem, get_loc(term, 0), 2);
    }
    APP => {
      collect(mem, ask_arg(mem, term, 0));
      collect(mem, ask_arg(mem, term, 1));
      clear(mem, get_loc(term, 0), 2);
    }
    SUP => {
      collect(mem, ask_arg(mem, term, 0));
      collect(mem, ask_arg(mem, term, 1));
      clear(mem, get_loc(term, 0), 2);
    }
    OP2 => {
      collect(mem, ask_arg(mem, term, 0));
      collect(mem, ask_arg(mem, term, 1));
    }
    U32 => {}
    CTR | CAL => {
      let arity = get_ari(term);
      for i in 0..arity {
        collect(mem, ask_arg(mem, term, i));
      }
      clear(mem, get_loc(term, 0), arity);
    }
    _ => {}
  }
}

pub fn inc_cost(mem: &mut Worker) {
  mem.cost += 1;
}

// Reduction
// ---------

// Performs a `x <- value` substitution. It just calls link if the substituted
// value is a term. If it is an ERA node, that means `value` is now unreachable,
// so we just call the collector.
pub fn subst(mem: &mut Worker, lnk: Lnk, val: Lnk) {
  if get_tag(lnk) != ERA {
    link(mem, get_loc(lnk, 0), val);
  } else {
    collect(mem, val);
  }
}

// (F {a0 a1} b c ...)
// ------------------- CAL-SUP
// dup b0 b1 = b
// dup c0 c1 = c
// ...
// {(F a0 b0 c0 ...) (F a1 b1 c1 ...)}
pub fn cal_sup(mem: &mut Worker, host: u64, term: Lnk, argn: Lnk, n: u64) -> Lnk {
  inc_cost(mem);
  let arit = get_ari(term);
  let func = get_ext(term);
  let fun0 = get_loc(term, 0);
  let fun1 = alloc(mem, arit);
  let sup0 = get_loc(argn, 0);
  for i in 0..arit {
    if i != n {
      let leti = alloc(mem, 3);
      let argi = ask_arg(mem, term, i);
      link(mem, fun0 + i, Dp0(get_ext(argn), leti));
      link(mem, fun1 + i, Dp1(get_ext(argn), leti));
      link(mem, leti + 2, argi);
    } else {
      link(mem, fun0 + i, ask_arg(mem, argn, 0));
      link(mem, fun1 + i, ask_arg(mem, argn, 1));
    }
  }
  link(mem, sup0 + 0, Cal(arit, func, fun0));
  link(mem, sup0 + 1, Cal(arit, func, fun1));
  let done = Sup(get_ext(argn), sup0);
  link(mem, host, done);
  done
}

// Reduces a term to weak head normal form.
pub fn reduce(
  mem: &mut Worker,
  funcs: &[Option<Function>],
  root: u64,
  _opt_id_to_name: Option<&HashMap<u64, String>>,
  debug: bool,
) -> Lnk {
  let mut stack: Vec<u64> = Vec::new();

  let mut init = 1;
  let mut host = root;

  loop {
    let term = ask_lnk(mem, host);

    if debug {
      println!("------------------------");
      println!("{}", show_term(mem, ask_lnk(mem, root), _opt_id_to_name, term));
    }

    if init == 1 {
      match get_tag(term) {
        APP => {
          stack.push(host);
          init = 1;
          host = get_loc(term, 0);
          continue;
        }
        DP0 | DP1 => {
          stack.push(host);
          host = get_loc(term, 2);
          continue;
        }
        OP2 => {
          stack.push(host);
          stack.push(get_loc(term, 0) | 0x80000000);
          host = get_loc(term, 1);
          continue;
        }
        CAL => {
          let fun = get_ext(term);
          let ari = get_ari(term);
          if let Some(f) = &funcs[fun as usize] {
            let len = f.stricts.len() as u64;
            if ari == f.arity {
              if len == 0 {
                init = 0;
              } else {
                stack.push(host);
                for (i, strict) in f.stricts.iter().enumerate() {
                  if i < f.stricts.len() - 1 {
                    stack.push(get_loc(term, *strict) | 0x80000000);
                  } else {
                    host = get_loc(term, *strict);
                  }
                }
              }
              continue;
            }
          }
        }
        _ => {}
      }
    } else {
      match get_tag(term) {
        APP => {
          let arg0 = ask_arg(mem, term, 0);
          match get_tag(arg0) {
            // (λx(body) a)
            // ------------ APP-LAM
            // x <- a
            // body
            LAM => {
              //println!("app-lam");
              inc_cost(mem);
              subst(mem, ask_arg(mem, arg0, 0), ask_arg(mem, term, 1));
              let _done = link(mem, host, ask_arg(mem, arg0, 1));
              clear(mem, get_loc(term, 0), 2);
              clear(mem, get_loc(arg0, 0), 2);
              init = 1;
              continue;
            }
            // ({a b} c)
            // ----------------- APP-SUP
            // dup x0 x1 = c
            // {(a x0) (b x1)}
            SUP => {
              //println!("app-sup");
              inc_cost(mem);
              let app0 = get_loc(term, 0);
              let app1 = get_loc(arg0, 0);
              let let0 = alloc(mem, 3);
              let sup0 = alloc(mem, 2);
              link(mem, let0 + 2, ask_arg(mem, term, 1));
              link(mem, app0 + 1, Dp0(get_ext(arg0), let0));
              link(mem, app0 + 0, ask_arg(mem, arg0, 0));
              link(mem, app1 + 0, ask_arg(mem, arg0, 1));
              link(mem, app1 + 1, Dp1(get_ext(arg0), let0));
              link(mem, sup0 + 0, App(app0));
              link(mem, sup0 + 1, App(app1));
              let done = Sup(get_ext(arg0), sup0);
              link(mem, host, done);
            }
            _ => {}
          }
        }
        DP0 | DP1 => {
          let arg0 = ask_arg(mem, term, 2);
          // let argK = ask_arg(mem, term, if get_tag(term) == DP0 { 1 } else { 0 });
          // if get_tag(argK) == ERA {
          //   let done = arg0;
          //   link(mem, host, done);
          //   init = 1;
          //   continue;
          // }
          match get_tag(arg0) {
            // dup r s = λx(f)
            // --------------- DUP-LAM
            // dup f0 f1 = f
            // r <- λx0(f0)
            // s <- λx1(f1)
            // x <- {x0 x1}
            LAM => {
              //println!("dup-lam");
              inc_cost(mem);
              let let0 = get_loc(term, 0);
              let sup0 = get_loc(arg0, 0);
              let lam0 = alloc(mem, 2);
              let lam1 = alloc(mem, 2);
              link(mem, let0 + 2, ask_arg(mem, arg0, 1));
              link(mem, sup0 + 1, Var(lam1));
              let arg0_arg_0 = ask_arg(mem, arg0, 0);
              link(mem, sup0 + 0, Var(lam0));
              subst(mem, arg0_arg_0, Sup(get_ext(term), sup0));
              let term_arg_0 = ask_arg(mem, term, 0);
              link(mem, lam0 + 1, Dp0(get_ext(term), let0));
              subst(mem, term_arg_0, Lam(lam0));
              let term_arg_1 = ask_arg(mem, term, 1);
              link(mem, lam1 + 1, Dp1(get_ext(term), let0));
              subst(mem, term_arg_1, Lam(lam1));
              let done = Lam(if get_tag(term) == DP0 { lam0 } else { lam1 });
              link(mem, host, done);
              init = 1;
              continue;
            }
            // dup x y = {a b}
            // --------------- DUP-SUP (equal)
            // x <- a
            // y <- b
            //
            // dup x y = {a b}
            // ----------------- DUP-SUP (different)
            // x <- {xA xB}
            // y <- {yA yB}
            // dup xA yA = a
            // dup xB yB = b
            SUP => {
              //println!("dup-sup");
              if get_ext(term) == get_ext(arg0) {
                inc_cost(mem);
                subst(mem, ask_arg(mem, term, 0), ask_arg(mem, arg0, 0));
                subst(mem, ask_arg(mem, term, 1), ask_arg(mem, arg0, 1));
                let _done =
                  link(mem, host, ask_arg(mem, arg0, if get_tag(term) == DP0 { 0 } else { 1 }));
                clear(mem, get_loc(term, 0), 3);
                clear(mem, get_loc(arg0, 0), 2);
                init = 1;
                continue;
              } else {
                inc_cost(mem);
                let sup0 = alloc(mem, 2);
                let let0 = get_loc(term, 0);
                let sup1 = get_loc(arg0, 0);
                let let1 = alloc(mem, 3);
                link(mem, let0 + 2, ask_arg(mem, arg0, 0));
                link(mem, let1 + 2, ask_arg(mem, arg0, 1));
                let term_arg_0 = ask_arg(mem, term, 0);
                let term_arg_1 = ask_arg(mem, term, 1);
                link(mem, sup1 + 0, Dp1(get_ext(term), let0));
                link(mem, sup1 + 1, Dp1(get_ext(term), let1));
                link(mem, sup0 + 0, Dp0(get_ext(term), let0));
                link(mem, sup0 + 1, Dp0(get_ext(term), let1));
                subst(mem, term_arg_0, Sup(get_ext(arg0), sup0));
                subst(mem, term_arg_1, Sup(get_ext(arg0), sup1));
                let done = Sup(get_ext(arg0), if get_tag(term) == DP0 { sup0 } else { sup1 });
                link(mem, host, done);
              }
            }
            // dup x y = N
            // ----------- DUP-U32
            // x <- N
            // y <- N
            // ~
            U32 => {
              //println!("dup-u32");
              inc_cost(mem);
              subst(mem, ask_arg(mem, term, 0), arg0);
              subst(mem, ask_arg(mem, term, 1), arg0);
              let _done = arg0;
              link(mem, host, arg0);
            }
            // dup x y = (K a b c ...)
            // ----------------------- DUP-CTR
            // dup a0 a1 = a
            // dup b0 b1 = b
            // dup c0 c1 = c
            // ...
            // x <- (K a0 b0 c0 ...)
            // y <- (K a1 b1 c1 ...)
            CTR => {
              //println!("dup-ctr");
              inc_cost(mem);
              let func = get_ext(arg0);
              let arit = get_ari(arg0);
              if arit == 0 {
                subst(mem, ask_arg(mem, term, 0), Ctr(0, func, 0));
                subst(mem, ask_arg(mem, term, 1), Ctr(0, func, 0));
                clear(mem, get_loc(term, 0), 3);
                let _done = link(mem, host, Ctr(0, func, 0));
              } else {
                let ctr0 = get_loc(arg0, 0);
                let ctr1 = alloc(mem, arit);
                for i in 0..arit - 1 {
                  let leti = alloc(mem, 3);
                  link(mem, leti + 2, ask_arg(mem, arg0, i));
                  link(mem, ctr0 + i, Dp0(get_ext(term), leti));
                  link(mem, ctr1 + i, Dp1(get_ext(term), leti));
                }
                let leti = get_loc(term, 0);
                link(mem, leti + 2, ask_arg(mem, arg0, arit - 1));
                let term_arg_0 = ask_arg(mem, term, 0);
                link(mem, ctr0 + arit - 1, Dp0(get_ext(term), leti));
                subst(mem, term_arg_0, Ctr(arit, func, ctr0));
                let term_arg_1 = ask_arg(mem, term, 1);
                link(mem, ctr1 + arit - 1, Dp1(get_ext(term), leti));
                subst(mem, term_arg_1, Ctr(arit, func, ctr1));
                let done = Ctr(arit, func, if get_tag(term) == DP0 { ctr0 } else { ctr1 });
                link(mem, host, done);
              }
            }
            _ => {}
          }
        }
        OP2 => {
          let arg0 = ask_arg(mem, term, 0);
          let arg1 = ask_arg(mem, term, 1);
          // (+ a b)
          // --------- OP2-U32
          // add(a, b)
          if get_tag(arg0) == U32 && get_tag(arg1) == U32 {
            //println!("op2-u32");
            inc_cost(mem);
            let a = get_val(arg0);
            let b = get_val(arg1);
            let c = match get_ext(term) {
              ADD => (a + b) & 0xFFFFFFFF,
              SUB => (a - b) & 0xFFFFFFFF,
              MUL => (a * b) & 0xFFFFFFFF,
              DIV => (a / b) & 0xFFFFFFFF,
              MOD => (a % b) & 0xFFFFFFFF,
              AND => (a & b) & 0xFFFFFFFF,
              OR => (a | b) & 0xFFFFFFFF,
              XOR => (a ^ b) & 0xFFFFFFFF,
              SHL => (a << b) & 0xFFFFFFFF,
              SHR => (a >> b) & 0xFFFFFFFF,
              LTN => u64::from(a < b),
              LTE => u64::from(a <= b),
              EQL => u64::from(a == b),
              GTE => u64::from(a >= b),
              GTN => u64::from(a > b),
              NEQ => u64::from(a != b),
              _ => 0,
            };
            let done = U_32(c);
            clear(mem, get_loc(term, 0), 2);
            link(mem, host, done);
          }
          // (+ {a0 a1} b)
          // --------------------- OP2-SUP-0
          // let b0 b1 = b
          // {(+ a0 b0) (+ a1 b1)}
          else if get_tag(arg0) == SUP {
            //println!("op2-sup-0");
            inc_cost(mem);
            let op20 = get_loc(term, 0);
            let op21 = get_loc(arg0, 0);
            let let0 = alloc(mem, 3);
            let sup0 = alloc(mem, 2);
            link(mem, let0 + 2, arg1);
            link(mem, op20 + 1, Dp0(get_ext(arg0), let0));
            link(mem, op20 + 0, ask_arg(mem, arg0, 0));
            link(mem, op21 + 0, ask_arg(mem, arg0, 1));
            link(mem, op21 + 1, Dp1(get_ext(arg0), let0));
            link(mem, sup0 + 0, Op2(get_ext(term), op20));
            link(mem, sup0 + 1, Op2(get_ext(term), op21));
            let done = Sup(get_ext(arg0), sup0);
            link(mem, host, done);
          }
          // (+ a {b0 b1})
          // --------------- OP2-SUP-1
          // dup a0 a1 = a
          // {(+ a0 b0) (+ a1 b1)}
          else if get_tag(arg1) == SUP {
            //println!("op2-sup-1");
            inc_cost(mem);
            let op20 = get_loc(term, 0);
            let op21 = get_loc(arg1, 0);
            let let0 = alloc(mem, 3);
            let sup0 = alloc(mem, 2);
            link(mem, let0 + 2, arg0);
            link(mem, op20 + 0, Dp0(get_ext(arg1), let0));
            link(mem, op20 + 1, ask_arg(mem, arg1, 0));
            link(mem, op21 + 1, ask_arg(mem, arg1, 1));
            link(mem, op21 + 0, Dp1(get_ext(arg1), let0));
            link(mem, sup0 + 0, Op2(get_ext(term), op20));
            link(mem, sup0 + 1, Op2(get_ext(term), op21));
            let done = Sup(get_ext(arg1), sup0);
            link(mem, host, done);
          }
        }
        CAL => {
          let fun = get_ext(term);
          let _ari = get_ari(term);
          if let Some(f) = &funcs[fun as usize] {
            if (f.rewriter)(mem, host, term) {
              //println!("cal-fun");
              init = 1;
              continue;
            }
          }
        }
        _ => {}
      }
    }

    if let Some(item) = stack.pop() {
      init = item >> 31;
      host = item & 0x7FFFFFFF;
      continue;
    }

    break;
  }

  ask_lnk(mem, root)
}

// sets the nth bit of a bit-array represented as a u64 array
pub fn set_bit(bits: &mut [u64], bit: u64) {
  bits[bit as usize >> 6] |= 1 << (bit & 0x3f);
}

// gets the nth bit of a bit-array represented as a u64 array
pub fn get_bit(bits: &[u64], bit: u64) -> bool {
  (((bits[bit as usize >> 6] >> (bit & 0x3f)) as u8) & 1) == 1
}

pub fn normal_go(
  mem: &mut Worker,
  funcs: &[Option<Function>],
  host: u64,
  seen: &mut [u64],
  opt_id_to_name: Option<&HashMap<u64, String>>,
  debug: bool,
) -> Lnk {
  let term = ask_lnk(mem, host);
  if get_bit(seen, host) {
    term
  } else {
    let term = reduce(mem, funcs, host, opt_id_to_name, debug);
    set_bit(seen, host);
    let mut rec_locs = Vec::with_capacity(16);
    match get_tag(term) {
      LAM => {
        rec_locs.push(get_loc(term, 1));
      }
      APP => {
        rec_locs.push(get_loc(term, 0));
        rec_locs.push(get_loc(term, 1));
      }
      SUP => {
        rec_locs.push(get_loc(term, 0));
        rec_locs.push(get_loc(term, 1));
      }
      DP0 => {
        rec_locs.push(get_loc(term, 2));
      }
      DP1 => {
        rec_locs.push(get_loc(term, 2));
      }
      CTR | CAL => {
        let arity = get_ari(term);
        for i in 0..arity {
          rec_locs.push(get_loc(term, i));
        }
      }
      _ => {}
    }
    for loc in rec_locs {
      let lnk: Lnk = normal_go(mem, funcs, loc, seen, opt_id_to_name, debug);
      link(mem, loc, lnk);
    }
    term
  }
}

pub fn normal(
  mem: &mut Worker,
  host: u64,
  funcs: &[Option<Function>],
  opt_id_to_name: Option<&HashMap<u64, String>>,
  debug: bool,
) -> Lnk {
  let mut seen = vec![0; 4194304];
  normal_go(mem, funcs, host, &mut seen, opt_id_to_name, debug)
}

// Debug
// -----

pub fn show_lnk(x: Lnk) -> String {
  if x == 0 {
    String::from("~")
  } else {
    let tag = get_tag(x);
    let ext = get_ext(x);
    let val = get_val(x);
    let ari = match tag {
      CTR => format!("{}", get_ari(x)),
      CAL => format!("{}", get_ari(x)),
      _ => String::new(),
    };
    let tgs = match tag {
      DP0 => "DP0",
      DP1 => "DP1",
      VAR => "VAR",
      ARG => "ARG",
      ERA => "ERA",
      LAM => "LAM",
      APP => "APP",
      SUP => "SUP",
      CTR => "CTR",
      CAL => "CAL",
      OP2 => "OP2",
      U32 => "U32",
      F32 => "F32",
      NIL => "NIL",
      _ => "???",
    };
    format!("{}{}:{:x}:{:x}", tgs, ari, ext, val)
  }
}

pub fn show_mem(worker: &Worker) -> String {
  let mut s: String = String::new();
  for i in 0..48 {
    // pushes to the string
    s.push_str(&format!("{:x} | ", i));
    s.push_str(&show_lnk(worker.node[i]));
    s.push('\n');
  }
  s
}

pub fn show_term(
  mem: &Worker,
  term: Lnk,
  opt_id_to_name: Option<&HashMap<u64, String>>,
  focus: u64,
) -> String {
  let mut lets: HashMap<u64, u64> = HashMap::new();
  let mut kinds: HashMap<u64, u64> = HashMap::new();
  let mut names: HashMap<u64, String> = HashMap::new();
  let mut count: u64 = 0;
  fn find_lets(
    mem: &Worker,
    term: Lnk,
    lets: &mut HashMap<u64, u64>,
    kinds: &mut HashMap<u64, u64>,
    names: &mut HashMap<u64, String>,
    count: &mut u64,
  ) {
    match get_tag(term) {
      LAM => {
        names.insert(get_loc(term, 0), format!("{}", count));
        *count += 1;
        find_lets(mem, ask_arg(mem, term, 1), lets, kinds, names, count);
      }
      APP => {
        find_lets(mem, ask_arg(mem, term, 0), lets, kinds, names, count);
        find_lets(mem, ask_arg(mem, term, 1), lets, kinds, names, count);
      }
      SUP => {
        find_lets(mem, ask_arg(mem, term, 0), lets, kinds, names, count);
        find_lets(mem, ask_arg(mem, term, 1), lets, kinds, names, count);
      }
      DP0 => {
        if let hash_map::Entry::Vacant(e) = lets.entry(get_loc(term, 0)) {
          names.insert(get_loc(term, 0), format!("{}", count));
          *count += 1;
          kinds.insert(get_loc(term, 0), get_ext(term));
          e.insert(get_loc(term, 0));
          find_lets(mem, ask_arg(mem, term, 2), lets, kinds, names, count);
        }
      }
      DP1 => {
        if let hash_map::Entry::Vacant(e) = lets.entry(get_loc(term, 0)) {
          names.insert(get_loc(term, 0), format!("{}", count));
          *count += 1;
          kinds.insert(get_loc(term, 0), get_ext(term));
          e.insert(get_loc(term, 0));
          find_lets(mem, ask_arg(mem, term, 2), lets, kinds, names, count);
        }
      }
      OP2 => {
        find_lets(mem, ask_arg(mem, term, 0), lets, kinds, names, count);
        find_lets(mem, ask_arg(mem, term, 1), lets, kinds, names, count);
      }
      CTR | CAL => {
        let arity = get_ari(term);
        for i in 0..arity {
          find_lets(mem, ask_arg(mem, term, i), lets, kinds, names, count);
        }
      }
      _ => {}
    }
  }
  fn go(
    mem: &Worker,
    term: Lnk,
    names: &HashMap<u64, String>,
    opt_id_to_name: Option<&HashMap<u64, String>>,
    focus: u64,
  ) -> String {
    let done = match get_tag(term) {
      DP0 => {
        format!("a{}", names.get(&get_loc(term, 0)).unwrap_or(&String::from("?")))
      }
      DP1 => {
        format!("b{}", names.get(&get_loc(term, 0)).unwrap_or(&String::from("?")))
      }
      VAR => {
        format!("x{}", names.get(&get_loc(term, 0)).unwrap_or(&String::from("?")))
      }
      LAM => {
        let name = format!("x{}", names.get(&get_loc(term, 0)).unwrap_or(&String::from("?")));
        format!("λ{} {}", name, go(mem, ask_arg(mem, term, 1), names, opt_id_to_name, focus))
      }
      APP => {
        let func = go(mem, ask_arg(mem, term, 0), names, opt_id_to_name, focus);
        let argm = go(mem, ask_arg(mem, term, 1), names, opt_id_to_name, focus);
        format!("({} {})", func, argm)
      }
      SUP => {
        //let kind = get_ext(term);
        let func = go(mem, ask_arg(mem, term, 0), names, opt_id_to_name, focus);
        let argm = go(mem, ask_arg(mem, term, 1), names, opt_id_to_name, focus);
        format!("{{{} {}}}", func, argm)
      }
      OP2 => {
        let oper = get_ext(term);
        let val0 = go(mem, ask_arg(mem, term, 0), names, opt_id_to_name, focus);
        let val1 = go(mem, ask_arg(mem, term, 1), names, opt_id_to_name, focus);
        let symb = match oper {
          0x00 => "+",
          0x01 => "-",
          0x02 => "*",
          0x03 => "/",
          0x04 => "%",
          0x05 => "&",
          0x06 => "|",
          0x07 => "^",
          0x08 => "<<",
          0x09 => ">>",
          0x10 => "<",
          0x11 => "<=",
          0x12 => "=",
          0x13 => ">=",
          0x14 => ">",
          0x15 => "!=",
          _ => "?",
        };
        format!("({} {} {})", symb, val0, val1)
      }
      U32 => {
        format!("{}", get_val(term))
      }
      CTR | CAL => {
        let func = get_ext(term);
        let arit = get_ari(term);
        let args: Vec<String> =
          (0..arit).map(|i| go(mem, ask_arg(mem, term, i), names, opt_id_to_name, focus)).collect();
        let name = if let Some(id_to_name) = opt_id_to_name {
          id_to_name.get(&func).unwrap_or(&String::from("?")).clone()
        } else {
          format!(
            "{}{}",
            if get_tag(term) < CAL { String::from("C") } else { String::from("F") },
            func
          )
        };
        format!("({}{})", name, args.iter().map(|x| format!(" {}", x)).collect::<String>())
      }
      _ => String::from("?"),
    };
    if term == focus {
      format!("${}", done)
    } else {
      done
    }
  }
  find_lets(mem, term, &mut lets, &mut kinds, &mut names, &mut count);
  let mut text = go(mem, term, &names, opt_id_to_name, focus);
  for (_key, pos) in lets {
    // todo: reverse
    let what = String::from("?");
    //let kind = kinds.get(&key).unwrap_or(&0);
    let name = names.get(&pos).unwrap_or(&what);
    let nam0 =
      if ask_lnk(mem, pos + 0) == Era() { String::from("*") } else { format!("a{}", name) };
    let nam1 =
      if ask_lnk(mem, pos + 1) == Era() { String::from("*") } else { format!("b{}", name) };
    text.push_str(&format!(
      "\ndup {} {} = {};",
      //kind,
      nam0,
      nam1,
      go(mem, ask_lnk(mem, pos + 2), &names, opt_id_to_name, focus)
    ));
  }
  text
}
