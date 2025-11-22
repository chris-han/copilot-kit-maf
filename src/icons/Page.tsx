import { File as LucideFile, LucideProps } from 'lucide-react';

const Page = ({ className, ...props }: LucideProps) => {
  return <LucideFile className={className} {...props} />;
};

export default Page;