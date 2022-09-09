logger = open("$run_dir/log.txt", "a")
import Base.print
import Base.println
using Format
function print(xs...)
    Base.print(stdout, xs...)
    Base.print(logger, xs...)
    flush(logger)
end
function println(xs...)
    Base.print(stdout, "$(now()): ")
    Base.print(logger, "$(now()): ")
    Base.println(stdout, xs...)
    Base.println(logger, xs...)
    flush(logger)
end
function println()
    Base.println(stdout)
    Base.println(logger)
    flush(logger)
end

macro print(args...)
    printfmt(stdout, args...)
    printfmt(logger, args...)
    flush(logger)
end
