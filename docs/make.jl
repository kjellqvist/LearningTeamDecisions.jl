using LearningTeamDecisions
using Documenter

DocMeta.setdocmeta!(LearningTeamDecisions, :DocTestSetup, :(using LearningTeamDecisions); recursive=true)

makedocs(;
    modules=[LearningTeamDecisions],
    authors="Olle Kjellqvist",
    repo="https://github.com/kjellqvist/LearningTeamDecisions.jl/blob/{commit}{path}#{line}",
    sitename="LearningTeamDecisions.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://kjellqvist.github.io/LearningTeamDecisions.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kjellqvist/LearningTeamDecisions.jl",
    devbranch="main",
)
