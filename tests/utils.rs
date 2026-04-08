// Copyright 2023 The Regents of the University of California
// Copyright 2024 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use fst_reader::*;

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
        FstHierarchyEntry::VhdlVarInfo {
            type_name,
            var_type,
            data_type,
        } => {
            format!("VHDL Var Info: {type_name}, {var_type:?}, {data_type:?}")
        }
        FstHierarchyEntry::Array {
            name,
            array_type,
            left,
            right,
        } => {
            let combined = ((*left as u64) << 32) | (*right as u64);
            format!("Array: {name} {array_type:?} {combined}")
        }
        FstHierarchyEntry::Pack {
            name,
            pack_type,
            value,
        } => {
            format!("Pack: {name} {pack_type:?} {value}")
        }
        FstHierarchyEntry::SVEnum {
            name,
            enum_type,
            value,
        } => {
            format!("SVEnum: {name} {enum_type:?} {value}")
        }
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
        FstScopeType::SvArray => "SvArray",
    };
    con.to_string()
}
