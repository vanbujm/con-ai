import { HfInference } from "@huggingface/inference"
import { config } from "dotenv"
import { resolve } from "node:path"
import { constitutions, system_chat } from "./data/constitution.json"
import { createWriteStream, WriteStream } from "node:fs"
import { sleep } from "../util/sleep.js"
import { parseTrainingData } from "./parseTrainingData.js"
import cliProgress from "cli-progress"
import chalk from "chalk"

config()

const ACCESS_TOKEN = process.env.HUGGING_FACE_ACCESS_TOKEN

const inference = new HfInference(ACCESS_TOKEN)

const MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

const createTrainingData = async (
  trainingData: {
    initialPrompt: string
    initialResponse: string
  }[],
  fileStream: WriteStream,
) => {
  console.info(chalk.blue("Creating new training data..."))

  const loadingBar = new cliProgress.SingleBar(
    {
      format: chalk.green(`Progress [{bar}] {percentage}% | ETA: {eta}s | {value}/{total}`),
    },
    cliProgress.Presets.shades_classic,
  )

  loadingBar.start(trainingData.length, 0)

  for (const { initialPrompt, initialResponse } of trainingData) {
    const { critic: criticPrompt, revision: revisionPrompt } =
      constitutions[Math.floor(Math.random() * constitutions.length)]

    const systemChatToUse = system_chat[Math.floor(Math.random() * system_chat.length)]

    const messages = [
      ...systemChatToUse,
      {
        role: "user",
        content: initialPrompt,
      },
      {
        role: "assistant",
        content: initialResponse,
      },
      {
        role: "user",
        content: criticPrompt,
      },
    ]

    await sleep(500)
    const criticResponse = await inference.chatCompletion({
      model: MODEL,
      messages,
      max_tokens: 200,
    })

    await sleep(500)

    const revisionResponse = await inference.chatCompletion({
      model: MODEL,
      messages: [
        ...messages,
        criticResponse.choices[0].message,
        {
          role: "user",
          content: revisionPrompt,
        },
      ],
      max_tokens: 200,
    })

    const data = {
      initialPrompt,
      initialResponse,
      criticPrompt,
      criticResponse: criticResponse.choices[0].message.content,
      revisionPrompt,
      revisionResponse: revisionResponse.choices[0].message.content,
    }

    fileStream.write(`${JSON.stringify(data)}\n`)

    loadingBar.increment()
  }

  loadingBar.stop()
  console.info(chalk.blue("Finished creating new training data."))
}

let isFinished = false

const main = async () => {
  const trainingData = await parseTrainingData()

  const fileStream = createWriteStream(resolve("data/processed.jsonl"), { flags: "a" })

  await new Promise((resolve) => fileStream.once("open", resolve))

  await createTrainingData(trainingData, fileStream)
  fileStream.end()
  isFinished = true
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
