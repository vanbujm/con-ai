import { HfInference } from "@huggingface/inference"
import { WriteStream, readFileSync } from "node:fs"
import chalk from "chalk"
import cliProgress from "cli-progress"
import { system_chat } from "./data/constitution.json"
import { sleep } from "../util/sleep.js"
import { config } from "dotenv"

config()

const ACCESS_TOKEN = process.env.HUGGING_FACE_ACCESS_TOKEN
const inference = new HfInference(ACCESS_TOKEN)
const MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

const jobCritics = readFileSync("./src/data-generation/data/job_perspective_critique_revision.jsonl", "utf-8")
  .split("\n")
  .map((line) => JSON.parse(line))

export const createSAITrainingData = async (
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
    const { critic: criticPrompt, revision: revisionPrompt } = jobCritics[Math.floor(Math.random() * jobCritics.length)]

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
          content: `${revisionPrompt}. Do not mention your profession or role. Simply change the response to meet any criteria stated.`,
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
