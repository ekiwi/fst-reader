// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use fst_native::*;

pub fn hierarchy_to_str(entry: &FstHierarchyEntry) -> String {
    match entry {
        FstHierarchyEntry::Scope {
            name,
            tpe,
            component,
        } => format!("Scope: {name} ({}) {component}", hierarchy_tpe_to_str(tpe)),
        FstHierarchyEntry::UpScope => "UpScope".to_string(),
        FstHierarchyEntry::Var { name, handle, .. } => format!("({handle}): {name}"),
        FstHierarchyEntry::AttributeEnd => "EndAttr".to_string(),
        FstHierarchyEntry::PathName { name, id } => format!("PathName: {id} -> {name}"),
        FstHierarchyEntry::SourceStem {
            is_instantiation,
            path_id,
            line,
        } => format!("SourceStem:: {is_instantiation}, {path_id}, {line}"),
        FstHierarchyEntry::Comment { string } => format!("Comment: {string}"),
        FstHierarchyEntry::EnumTable {
            name,
            handle,
            mapping,
        } => {
            let names = mapping
                .iter()
                .map(|(_v, n)| n.clone())
                .collect::<Vec<_>>()
                .join(" ");
            let values = mapping
                .iter()
                .map(|(v, _n)| v.clone())
                .collect::<Vec<_>>()
                .join(" ");
            format!(
                "EnumTable: {name} {} {names} {values} ({handle})",
                mapping.len()
            )
        }
        FstHierarchyEntry::EnumTableRef { handle } => format!("EnumTableRef: {handle}"),
    }
}

fn hierarchy_tpe_to_str(tpe: &FstScopeType) -> String {
    let con = match tpe {
        FstScopeType::Module => "Module",
        FstScopeType::Task => "Task",
        FstScopeType::Function => "Function",
        FstScopeType::Begin => "Begin",
        FstScopeType::Fork => "Fork",
        FstScopeType::Generate => "Generate",
        FstScopeType::Struct => "Struct",
        FstScopeType::Union => "Union",
        FstScopeType::Class => "Class",
        FstScopeType::Interface => "Interface",
        FstScopeType::Package => "Package",
        FstScopeType::Program => "Program",
        FstScopeType::VhdlArchitecture => "VhdlArchitecture",
        FstScopeType::VhdlProcedure => "VhdlProcedure",
        FstScopeType::VhdlFunction => "VhdlFunction",
        FstScopeType::VhdlRecord => "VhdlRecord",
        FstScopeType::VhdlProcess => "VhdlProcess",
        FstScopeType::VhdlBlock => "VhdlBlock",
        FstScopeType::VhdlForGenerate => "VhdlForGenerate",
        FstScopeType::VhdlIfGenerate => "VhdlIfGenerate",
        FstScopeType::VhdlGenerate => "VhdlGenerate",
        FstScopeType::VhdlPackage => "VhdlPackage",
        FstScopeType::AttributeBegin => "AttributeBegin",
        FstScopeType::AttributeEnd => "AttributeEnd",
        FstScopeType::VcdScope => "VcdScope",
        FstScopeType::VcdUpScope => "VcdUpScope",
    };
    con.to_string()
}
