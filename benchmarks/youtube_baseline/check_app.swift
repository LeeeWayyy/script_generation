// Compile with the unchanged ScriptViewing App/Models.swift and
// App/TranscriptPresentation.swift. This checks app compatibility, not hearing.
import Foundation

@main struct CheckBaseline {
    static func main() throws {
        var results: [[String: Any]] = []
        for path in CommandLine.arguments.dropFirst() {
            var result: [String: Any] = ["file": URL(fileURLWithPath: path).lastPathComponent]
            do {
                let data = try Data(contentsOf: URL(fileURLWithPath: path))
                let rows = try MaterialParser.transcript(data)
                var cues = 0, entries = 0, noCues: [Int] = []
                for (index, row) in rows.enumerated() {
                    let timing = TranscriptWordCues(row)
                    entries += row.transcriptWords?.count ?? 0
                    if timing.cues.isEmpty { noCues.append(index) }
                    for cue in timing.cues {
                        guard cue.span.range(in: row.text) != nil,
                              timing.span(at: (cue.start + cue.end) / 2) == cue.span else {
                            throw StudyError("Word cue span/midpoint mismatch")
                        }
                        cues += 1
                    }
                }
                result.merge(["app_parser_pass": true, "rows": rows.count,
                              "aligned_entries": entries, "usable_cues": cues,
                              "rows_without_cues": noCues]) { _, new in new }
            } catch {
                result["app_parser_pass"] = false
                result["error"] = String(describing: error)
            }
            results.append(result)
        }
        let output = try JSONSerialization.data(withJSONObject: results, options: [.prettyPrinted, .sortedKeys])
        FileHandle.standardOutput.write(output)
    }
}
