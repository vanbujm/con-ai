import { resolve } from "path"
import { readFile, writeFile } from "node:fs/promises"

const main = async () => {
  const processedFile = await readFile(resolve("data/processed.jsonl"), "utf-8")

  const lines = processedFile.split("\n")

  const dedupedLines = Array.from(new Set(lines))

  console.log(`Removed ${lines.length - dedupedLines.length} duplicates`)

  await writeFile(resolve("data/processed-deduped.jsonl"), dedupedLines.join("\n"))
}

main()
