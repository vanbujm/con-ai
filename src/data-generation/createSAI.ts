import { resolve } from "node:path"
import { createWriteStream } from "node:fs"
import { sleep } from "../util/sleep.js"
import { parseTrainingData } from "./parseTrainingData.js"
import chalk from "chalk"
import { createSAITrainingData } from "./createSAITrainingData.js"

let isFinished = false

const main = async () => {
  const dataFile = resolve("data/sai-processed.jsonl")
  const trainingData = await parseTrainingData(dataFile)

  const fileStream = createWriteStream(dataFile, { flags: "a" })

  await new Promise((resolve) => fileStream.once("open", resolve))

  try {
    await createSAITrainingData(trainingData, fileStream)
    isFinished = true
  } finally {
    fileStream.end()
    process.exit(0)
  }
}

const runMain = async () => {
  let isRunning = false
  while (!isFinished) {
    try {
      if (!isRunning) {
        isRunning = true
        await main()
        isRunning = false
      }
    } catch (e) {
      console.warn(chalk.yellow(`\n${e}`))
    } finally {
      isRunning = false
      console.info(chalk.blue(`${new Date().toLocaleString()} : Sleeping for 10 minutes...`))
      await sleep(1000 * 60 * 10)
    }
  }
}

runMain()
