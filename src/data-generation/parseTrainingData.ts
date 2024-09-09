import { readFile } from "node:fs/promises"
import { resolve } from "node:path"
import cliProgress from "cli-progress"
import chalk from "chalk"

export const parseTrainingData = async () => {
  console.info(chalk.blue("Parsing initial training data..."))

  let numberOfLines = 0
  try {
    const dataFile = await readFile(resolve("data/processed.jsonl"), "utf-8")
    numberOfLines = dataFile.split("\n").length - 1
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
  } catch (e) {
    console.warn(chalk.yellow("Could not find processed data file."))
  }

  if (numberOfLines > 0) {
    console.info(chalk.blue(`${numberOfLines} lines already exist. Resuming from last point...`))
  }

  const loadingBar = new cliProgress.SingleBar(
    {
      format: chalk.green(`Progress [{bar}] {percentage}% | ETA: {eta}s | {value}/{total}`),
    },
    cliProgress.Presets.shades_classic,
  )

  const trainingDataRaw = await readFile(resolve("src/data-generation/data/train.jsonl"), "utf-8")

  const data = trainingDataRaw
    .split("\n")
    .map<
      | {
          initialPrompt: string
          initialResponse: string
        }
      | undefined
    >((line, index, arr) => {
      if (!loadingBar.isActive) {
        loadingBar.start(arr.length, 0)
      }

      if (line == null || line === "") return

      if (index + 1 < numberOfLines) {
        loadingBar.increment()
        return
      }

      let parsedLine: { rejected: string }

      try {
        parsedLine = JSON.parse(line) as { rejected: string }
      } catch (e) {
        console.info(`Error parsing line: ${index + 1}`)
        console.info(line)
        console.error(e)
      }

      // @ts-expect-error it might not be defined yet
      if (!parsedLine) return

      const rejected = parsedLine.rejected

      const split = rejected.split("Human:").filter((line) => {
        const match = /^(\s|\n)+$/.test(line)
        return !match
      })[0]

      const lineSplit = split.split("Assistant:").map((line) => line.replace(/\n+/g, "\n").trim())
      const human = lineSplit[0]
      const assistant = lineSplit[1]

      loadingBar.increment()
      return { initialPrompt: human, initialResponse: assistant }
    })
    .filter((line) => line != null)

  loadingBar.stop()

  console.info(chalk.blue("Finished parsing initial training data."))

  return data
}
